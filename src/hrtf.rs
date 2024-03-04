use std::path::Path;

use hrtf::HrirSphere;

use crate::hacker::{An, AudioNode, U1, U2};
use crate::linear_algebra::mat3::Mat3;
use crate::linear_algebra::vec3::Vec3;

#[derive(Clone)]
pub struct Listener {
    basis: Mat3,
    position: Vec3,
}

impl Listener {
    pub(crate) fn new() -> Self {
        Self {
            basis: Default::default(),
            position: Default::default(),
        }
    }

    /// Sets new basis from given vectors in left-handed coordinate system.
    /// See `set_basis` for more info.
    pub fn set_orientation_lh(&mut self, look: Vec3, up: Vec3) {
        self.basis = Mat3::from_vectors(look.cross(&up), up, look)
    }

    /// Sets new basis from given vectors in right-handed coordinate system.
    /// See `set_basis` for more info.
    pub fn set_orientation_rh(&mut self, look: Vec3, up: Vec3) {
        self.basis = Mat3::from_vectors(up.cross(&look), up, look)
    }

    /// Sets arbitrary basis. Basis defines orientation of the listener in space.
    /// In your application you can take basis of camera in world coordinates and
    /// pass it to this method. If you using HRTF, make sure your basis is in
    /// right-handed coordinate system! You can make fake right-handed basis from
    /// left handed, by inverting Z axis. It is fake because it will work only for
    /// positions (engine interested in positions only), but not for rotation, shear
    /// etc.
    ///
    /// # Notes
    ///
    /// Basis must have mutually perpendicular axes.
    ///
    /// ```
    /// use rg3d_sound::listener::Listener;
    /// use rg3d_sound::math::mat3::Mat3;
    /// use rg3d_sound::math::vec3::Vec3;
    /// use rg3d_sound::math::quat::Quat;
    ///
    /// fn orient_listener(listener: &mut Listener) {
    ///     let basis = Mat3::from_quat(Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 45.0f32.to_radians()));
    ///     listener.set_basis(basis);
    /// }
    /// ```
    pub fn set_basis(&mut self, matrix: Mat3) {
        self.basis = matrix;
    }

    /// Returns shared reference to current basis.
    pub fn basis(&self) -> &Mat3 {
        &self.basis
    }

    /// Sets current position in world space.
    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    /// Returns position of listener.
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Returns up axis from basis.
    pub fn up_axis(&self) -> Vec3 {
        self.basis.up()
    }

    /// Returns look axis from basis.
    pub fn look_axis(&self) -> Vec3 {
        self.basis.look()
    }

    /// Returns ear axis from basis.
    pub fn ear_axis(&self) -> Vec3 {
        self.basis.side()
    }
}

/// See module docs.
#[derive(Clone)]
pub struct Hrtf {
    processor: hrtf::HrtfProcessor,
    listener: Listener,
    distance_model: DistanceModel,
    position: Vec3,
    // For sounds eminating from a spherical area
    radius: f32,
    rolloff_factor: f32,
    max_distance: f32,
    // Some data that needed for iterative overlap-save convolution.
    pub(crate) prev_left_samples: Vec<f32>,
    pub(crate) prev_right_samples: Vec<f32>,
    pub(crate) prev_sampling_vector: Vec3,
    pub(crate) prev_distance_gain: Option<f32>,
    buffer: Vec<f32>,
    output_buffer: Vec<(f32, f32)>,
    filled_to: usize,
}

impl Hrtf {
    /// Creates new HRTF renderer using specified HRTF sphere. See module docs for more info.
    pub fn new(hrir_sphere: hrtf::HrirSphere, position: Vec3) -> Self {
        let res = Self {
            processor: hrtf::HrtfProcessor::new(
                hrir_sphere,
                HRTF_INTERPOLATION_STEPS,
                HRTF_BLOCK_LEN,
            ),
            listener: Listener::new(),
            distance_model: DistanceModel::LinearDistance,
            position,
            buffer: vec![0.; HRTF_BLOCK_LEN * HRTF_INTERPOLATION_STEPS],
            output_buffer: vec![(0., 0.); HRTF_BLOCK_LEN * HRTF_INTERPOLATION_STEPS],
            filled_to: 0,
            radius: 1.0,
            max_distance: std::f32::MAX,
            rolloff_factor: 1.0,
            prev_left_samples: vec![0.; 511],
            prev_right_samples: vec![0.; 511],
            prev_sampling_vector: Vec3::new(0.0, 0.0, 1.0),
            prev_distance_gain: None,
        };
        eprintln!("Sphere_length: {}", res.processor.hrtf_sphere().length - 1);
        res
    }

    // Distance models were taken from OpenAL Specification because it looks like they're
    // standard in industry and there is no need to reinvent it.
    // https://www.openal.org/documentation/openal-1.1-specification.pdf
    pub(crate) fn get_distance_gain(&self) -> f32 {
        let listener = &self.listener;
        let distance_model = &self.distance_model;
        let distance = self
            .position
            .distance(&listener.position())
            .max(self.radius)
            .min(self.max_distance);
        match distance_model {
            DistanceModel::None => 1.0,
            DistanceModel::InverseDistance => {
                self.radius / (self.radius + self.rolloff_factor * (distance - self.radius))
            }
            DistanceModel::LinearDistance => {
                1.0 - self.radius * (distance - self.radius) / (self.max_distance - self.radius)
            }
            DistanceModel::ExponentDistance => (distance / self.radius).powf(-self.rolloff_factor),
        }
    }

    pub fn get_panning(&self) -> f32 {
        let listener = &self.listener;
        (self.position - listener.position())
            .normalized()
            // Fallback to look axis will give zero panning which will result in even
            // gain in each channels (as if there was no panning at all).
            .unwrap_or_else(|| listener.look_axis())
            .dot(&listener.ear_axis())
    }

    pub(crate) fn get_sampling_vector(&self) -> Vec3 {
        self.listener
            .basis()
            .transform_vector(self.position - self.listener.position())
            .normalized()
            // This is ok to fallback to (0, 0, 1) vector because it's given
            // in listener coordinate system.
            .unwrap_or_else(|| Vec3::new(0.0, 0.0, 1.0))
    }

    /// fills the output buffer with new audio data derived from the input buffer
    // #panics if self.fill_to is not at the end of the output buffer
    fn process_block(&mut self) {
        // This is only called in one place but good to defend against future changes
        assert!(self.filled_to == self.buffer.len());
        let new_distance_gain = self.get_distance_gain();
        let new_sampling_vector = self.get_sampling_vector();

        self.processor.process_samples(hrtf::HrtfContext {
            source: &self.buffer,
            output: &mut self.output_buffer,
            new_sample_vector: new_sampling_vector.into(),
            prev_sample_vector: self.prev_sampling_vector.into(),
            prev_left_samples: &mut self.prev_left_samples,
            prev_right_samples: &mut self.prev_right_samples,
            prev_distance_gain: self.prev_distance_gain.unwrap_or(new_distance_gain),
            new_distance_gain,
        });

        self.prev_sampling_vector = new_sampling_vector;
        self.prev_distance_gain = Some(new_distance_gain);
    }
}

impl AudioNode for Hrtf {
    const ID: u64 = 40308;
    type Sample = f32;
    type Inputs = U1;
    type Outputs = U2;

    type Setting = Vec3;

    fn set(&mut self, setting: Self::Setting) {
        self.position = setting
    }

    fn tick(
        &mut self,
        input: &crate::hacker::Frame<Self::Sample, Self::Inputs>,
    ) -> crate::hacker::Frame<Self::Sample, Self::Outputs> {
        self.buffer[self.filled_to] = input[0];
        self.filled_to += 1;
        if self.filled_to == self.buffer.len() {
            self.process_block();
            self.buffer.fill(0.);
            self.filled_to = 0;
        }
        let output = self.output_buffer[self.filled_to];
        [output.0, output.1].into()
    }
}

/// TODO: This is magic constant that gives 1024 + 1 number when summed with
///       HRTF length for faster FFT calculations. Find a better way of selecting this.
pub const HRTF_BLOCK_LEN: usize = 513;
pub(crate) const HRTF_INTERPOLATION_STEPS: usize = 8;

/// Distance model defines how volume of sound will decay when distance to listener changes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DistanceModel {
    /// No distance attenuation at all.
    None,

    /// Distance will decay using following formula:
    ///
    /// `clamped_distance = min(max(distance, radius), max_distance)`
    /// `attenuation = radius / (radius + rolloff_factor * (clamped_distance - radius))`
    ///
    /// where - `radius` - of source at which it has maximum volume,
    ///         `max_distance` - distance at which decay will stop,
    ///         `rolloff_factor` - coefficient that defines how fast volume will decay
    ///
    /// # Notes
    ///
    /// This is default distance model of context.
    InverseDistance,

    /// Distance will decay using following formula:
    ///
    /// `clamped_distance = min(max(distance, radius), max_distance)`
    /// `attenuation = 1.0 - radius * (clamped_distance - radius) / (max_distance - radius)`
    ///
    /// where - `radius` - of source at which it has maximum volume,
    ///         `max_distance` - distance at which decay will stop
    ///
    /// # Notes
    ///
    /// As you can see `rolloff_factor` is ignored here because of linear law.
    LinearDistance,

    /// Distance will decay using following formula:
    ///
    /// `clamped_distance = min(max(distance, radius), max_distance)`
    /// `(clamped_distance / radius) ^ (-rolloff_factor)`
    ///
    /// where - `radius` - of source at which it has maximum volume,
    ///         `max_distance` - distance at which decay will stop,
    ///         `rolloff_factor` - coefficient that defines how fast volume will decay
    ExponentDistance,
}

impl From<Vec3> for hrtf::Vec3 {
    fn from(val: Vec3) -> Self {
        hrtf::Vec3 {
            x: val.x,
            y: val.y,
            z: val.z,
        }
    }
}

pub fn hrtf<P: AsRef<Path>>(path: P, sample_rate: u32) -> An<Hrtf> {
    let sphere = HrirSphere::from_file(path, sample_rate).unwrap();
    An(Hrtf::new(sphere, Vec3::new(0., 0., 10.)))
}
