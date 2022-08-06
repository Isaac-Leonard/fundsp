//! Multichannel wave abstraction.

use super::audionode::*;
use super::audiounit::*;
use super::math::*;
use super::*;
use duplicate::duplicate_item;
use numeric_array::*;
use rsor::Slice;
use std::fs::File;
use std::io::prelude::*;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

/// Write a 32-bit value to a WAV file.
#[inline]
fn write32<W: Write>(writer: &mut W, x: u32) -> std::io::Result<()> {
    // WAV files are little endian.
    writer.write_all(&[x as u8, (x >> 8) as u8, (x >> 16) as u8, (x >> 24) as u8])?;
    std::io::Result::Ok(())
}

/// Write a 16-bit value to a WAV file.
#[inline]
fn write16<W: Write>(writer: &mut W, x: u16) -> std::io::Result<()> {
    writer.write_all(&[x as u8, (x >> 8) as u8])?;
    std::io::Result::Ok(())
}

// Write WAV header, including the header of the data block.
fn write_wav_header<W: Write>(
    writer: &mut W,
    data_length: usize,
    format: u16,
    channels: usize,
    sample_rate: usize,
) -> std::io::Result<()> {
    writer.write_all(b"RIFF")?;
    write32(writer, data_length as u32 + 36)?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    // Length of fmt block.
    write32(writer, 16)?;
    // Audio data format 1 = WAVE_FORMAT_PCM, 3 = WAVE_FORMAT_IEEE_FLOAT.
    write16(writer, format)?;
    write16(writer, channels as u16)?;
    write32(writer, sample_rate as u32)?;
    // Data rate in bytes per second.
    let sample_bytes = if format == 1 { 2 } else { 4 };
    write32(writer, (sample_rate * channels) as u32 * sample_bytes)?;
    // Sample frame length in bytes.
    write16(writer, channels as u16 * 2)?;
    // Bits per sample.
    write16(writer, 16)?;
    writer.write_all(b"data")?;
    // Length of data block.
    write32(writer, data_length as u32)?;
    std::io::Result::Ok(())
}

/// Multichannel wave.
#[duplicate_item(
    f48       Wave48       AudioUnit48;
    [ f64 ]   [ Wave64 ]   [ AudioUnit64 ];
    [ f32 ]   [ Wave32 ]   [ AudioUnit32 ];
)]
pub struct Wave48 {
    /// Vector of channels. Each channel is stored in its own vector.
    vec: Vec<Vec<f48>>,
    /// Sample rate of the wave.
    sr: f64,
    /// Slice of references. This is only allocated if it is used.
    slice: Slice<[f48]>,
}

#[duplicate_item(
    f48       Wave48       AudioUnit48;
    [ f64 ]   [ Wave64 ]   [ AudioUnit64 ];
    [ f32 ]   [ Wave32 ]   [ AudioUnit32 ];
)]
impl Wave48 {
    /// Creates an empty wave with the specified number of channels (`channels` > 0).
    pub fn new(channels: usize, sample_rate: f64) -> Self {
        assert!(channels > 0);
        let mut vec = Vec::with_capacity(channels);
        for _i in 0..channels {
            vec.push(Vec::new());
        }
        Self {
            vec,
            sr: sample_rate,
            slice: Slice::new(),
        }
    }

    /// Creates an empty wave with the given `capacity` in samples
    /// and number of channels (`channels` > 0).
    pub fn with_capacity(channels: usize, sample_rate: f64, capacity: usize) -> Self {
        assert!(channels > 0);
        let mut vec = Vec::with_capacity(channels);
        for _i in 0..channels {
            vec.push(Vec::with_capacity(capacity));
        }
        Self {
            vec,
            sr: sample_rate,
            slice: Slice::new(),
        }
    }

    /// Sample rate of this wave.
    pub fn sample_rate(&self) -> f64 {
        self.sr
    }

    /// Set the sample rate.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sr = sample_rate;
    }

    /// Number of channels in this wave.
    pub fn channels(&self) -> usize {
        self.vec.len()
    }

    /// Return a reference to the requested channel.
    pub fn channel(&self, channel: usize) -> &Vec<f48> {
        &self.vec[channel]
    }

    /// Return a mutable reference to the requested channel.
    pub fn channel_mut(&mut self, channel: usize) -> &mut Vec<f48> {
        &mut self.vec[channel]
    }

    /// Return a reference to the channels vector as a slice of slices.
    pub fn channels_ref(&mut self) -> &[&[f48]] {
        self.slice.from_refs(&self.vec)
    }

    /// Return a mutable reference to the channels vector as a slice of slices.
    pub fn channels_mut(&mut self) -> &mut [&mut [f48]] {
        self.slice.from_muts(&mut self.vec)
    }

    /// Sample accessor.
    pub fn at(&self, channel: usize, index: usize) -> f48 {
        self.vec[channel][index]
    }

    /// Set sample to value.
    pub fn set(&mut self, channel: usize, index: usize, value: f48) {
        self.vec[channel][index] = value;
    }

    /// Length of the wave in samples.
    pub fn length(&self) -> usize {
        self.vec[0].len()
    }

    /// Length of the wave in samples.
    pub fn len(&self) -> usize {
        self.vec[0].len()
    }

    /// Returns whether this wave contains no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Duration of the wave in seconds.
    pub fn duration(&self) -> f64 {
        self.length() as f64 / self.sample_rate()
    }

    /// Resizes the wave in-place. Any new samples are set to zero.
    pub fn resize(&mut self, length: usize) {
        if length != self.length() {
            for channel in 0..self.channels() {
                self.vec[channel].resize(length, 0.0);
            }
        }
    }

    /// Peak amplitude of the wave.
    pub fn amplitude(&self) -> f48 {
        let mut peak = 0.0;
        for channel in 0..self.channels() {
            for i in 0..self.len() {
                peak = max(peak, abs(self.at(channel, i)));
            }
        }
        peak
    }

    /// Scales the wave to the range -1...1.
    pub fn normalize(&mut self) {
        let a = self.amplitude();
        if a == 0.0 || a == 1.0 {
            return;
        }
        let z = 1.0 / a;
        for channel in 0..self.channels() {
            for i in 0..self.len() {
                self.set(channel, i, self.at(channel, i) * z);
            }
        }
    }

    /// Render wave with length `duration` seconds from a generator `node`.
    /// Resets `node` and sets its sample rate.
    /// Does not discard pre-delay.
    pub fn render(sample_rate: f64, duration: f64, node: &mut dyn AudioUnit48) -> Self {
        assert!(node.inputs() == 0);
        assert!(node.outputs() > 0);
        node.reset(Some(sample_rate));
        let length = (duration * sample_rate).round() as usize;
        let mut wave = Self::with_capacity(node.outputs(), sample_rate, length);
        let mut i = 0;
        let mut buffer = Self::new(node.outputs(), sample_rate);
        let mut reusable_slice = Slice::<[f48]>::with_capacity(node.outputs());
        while i < length {
            let n = min(length - i, MAX_BUFFER_SIZE);
            buffer.resize(n);
            node.process(n, &[], reusable_slice.from_muts(&mut buffer.vec));
            for channel in 0..node.outputs() {
                wave.vec[channel].extend_from_slice(&buffer.vec[channel][..]);
            }
            i += n;
        }
        wave
    }

    /// Render wave with length `duration` seconds from a generator `node`.
    /// Any pre-delay, as measured by signal latency, is discarded.
    /// Resets `node` and sets its sample rate.
    pub fn render_latency(sample_rate: f64, duration: f64, node: &mut dyn AudioUnit48) -> Self {
        assert!(node.inputs() == 0);
        assert!(node.outputs() > 0);
        let latency = node.latency().unwrap_or_default();
        // Round latency down to nearest sample.
        let latency_samples = floor(latency) as usize;
        let latency_duration = latency_samples as f64 / sample_rate;
        // Round duration to nearest sample.
        let duration_samples = round(duration * sample_rate) as usize;
        let duration = duration_samples as f64 / sample_rate;
        if latency_samples > 0 {
            let latency_wave = Self::render(sample_rate, duration + latency_duration, node);
            let mut wave = Self::with_capacity(node.outputs(), sample_rate, duration_samples);
            wave.resize(duration_samples);
            for channel in 0..wave.channels() {
                for i in 0..duration_samples {
                    wave.set(channel, i, latency_wave.at(channel, i + latency_samples));
                }
            }
            wave
        } else {
            Self::render(sample_rate, duration, node)
        }
    }

    /// Filter this wave with `node` and return the resulting wave.
    /// Resets `node` and sets its sample rate. Does not discard pre-delay.
    /// The `node` must have as many inputs as there are channels in this wave.
    /// All zeros input is used for the rest of the wave if
    /// the duration is greater than the duration of this wave.
    pub fn filter(&self, duration: f64, node: &mut dyn AudioUnit48) -> Self {
        assert!(node.inputs() == self.channels());
        assert!(node.outputs() > 0);
        node.reset(Some(self.sample_rate()));
        let total_length = round(duration * self.sample_rate()) as usize;
        let input_length = min(total_length, self.length());
        let mut wave = Self::with_capacity(node.outputs(), self.sample_rate(), total_length);
        let mut i = 0;
        let mut input_buffer = Self::new(self.channels(), self.sample_rate());
        let mut reusable_input_slice = Slice::<[f48]>::with_capacity(self.channels());
        let mut output_buffer = Self::new(node.outputs(), self.sample_rate());
        let mut reusable_output_slice = Slice::<[f48]>::with_capacity(node.outputs());
        // Filter from this wave.
        while i < input_length {
            let n = min(input_length - i, MAX_BUFFER_SIZE);
            input_buffer.resize(n);
            output_buffer.resize(n);
            for channel in 0..self.channels() {
                for j in 0..n {
                    input_buffer.set(channel, j, self.at(channel, i + j));
                }
            }
            node.process(
                n,
                reusable_input_slice.from_refs(&input_buffer.vec),
                reusable_output_slice.from_muts(&mut output_buffer.vec),
            );
            for channel in 0..node.outputs() {
                wave.vec[channel].extend_from_slice(&output_buffer.vec[channel][..]);
            }
            i += n;
        }
        // Filter the rest from a zero input.
        if i < total_length {
            input_buffer.resize(MAX_BUFFER_SIZE);
            for channel in 0..self.channels() {
                for j in 0..MAX_BUFFER_SIZE {
                    input_buffer.set(channel, j, 0.0);
                }
            }
            while i < total_length {
                let n = min(total_length - i, MAX_BUFFER_SIZE);
                input_buffer.resize(n);
                output_buffer.resize(n);
                node.process(
                    n,
                    reusable_input_slice.from_refs(&input_buffer.vec),
                    reusable_output_slice.from_muts(&mut output_buffer.vec),
                );
                for channel in 0..node.outputs() {
                    wave.vec[channel].extend_from_slice(&output_buffer.vec[channel][..]);
                }
                i += n;
            }
        }
        wave
    }

    /// Filter this wave with `node` and return the resulting wave.
    /// Any pre-delay, as measured by signal latency, is discarded.
    /// Resets `node` and sets its sample rate.
    /// The `node` must have as many inputs as there are channels in this wave.
    /// All zeros input is used for the rest of the wave if
    /// the duration is greater than the duration of this wave.
    pub fn filter_latency(&self, duration: f64, node: &mut dyn AudioUnit48) -> Self {
        assert!(node.inputs() == self.channels());
        assert!(node.outputs() > 0);
        let latency = node.latency().unwrap_or_default();
        // Round latency down to nearest sample.
        let latency_samples = floor(latency) as usize;
        let latency_duration = latency_samples as f64 / self.sample_rate();
        // Round duration to nearest sample.
        let duration_samples = round(duration * self.sample_rate()) as usize;
        let duration = duration_samples as f64 / self.sample_rate();
        if latency_samples > 0 {
            let latency_wave = self.filter(duration + latency_duration, node);
            let mut wave =
                Self::with_capacity(node.outputs(), self.sample_rate(), duration_samples);
            wave.resize(duration_samples);
            for channel in 0..wave.channels() {
                for i in 0..duration_samples {
                    wave.set(channel, i, latency_wave.at(channel, i + latency_samples));
                }
            }
            wave
        } else {
            self.filter(duration, node)
        }
    }

    /// Writes the wave as a 16-bit WAV to a buffer.
    /// Individual samples are clipped to the range -1...1.
    pub fn write_wav16<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_wav_header(
            writer,
            2 * self.channels() * self.length(),
            1,
            self.channels(),
            round(self.sample_rate()) as usize,
        )?;
        for i in 0..self.length() {
            for channel in 0..self.channels() {
                let sample = round(clamp11(self.at(channel, i)) * 32767.49);
                write16(writer, sample.to_i64() as u16)?;
            }
        }
        std::io::Result::Ok(())
    }

    /// Writes the wave as a 32-bit float WAV to a buffer.
    /// Samples are not clipped to any range but some
    /// applications may expect the range to be -1...1.
    pub fn write_wav32<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_wav_header(
            writer,
            4 * self.channels() * self.length(),
            3,
            self.channels(),
            round(self.sample_rate()) as usize,
        )?;
        for i in 0..self.length() {
            for channel in 0..self.channels() {
                let sample = self.at(channel, i);
                writer.write_all(&sample.to_f32().to_le_bytes())?;
            }
        }
        std::io::Result::Ok(())
    }

    /// Saves the wave as a 16-bit WAV file.
    /// Individual samples are clipped to the range -1...1.
    pub fn save_wav16(&self, path: &Path) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        self.write_wav16(&mut file)
    }

    /// Saves the wave as a 32-bit float WAV file.
    /// Samples are not clipped to any range but some
    /// applications may expect the range to be -1...1.
    pub fn save_wav32(&self, path: &Path) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        self.write_wav32(&mut file)
    }
}

/// Play back one channel of a wave.
#[duplicate_item(
    f48       Wave48       Wave48Player;
    [ f64 ]   [ Wave64 ]   [ Wave64Player ];
    [ f32 ]   [ Wave32 ]   [ Wave32Player ];
)]
pub struct Wave48Player<T: Float> {
    wave: Arc<Wave48>,
    channel: usize,
    index: usize,
    loop_point: Option<usize>,
    _marker: PhantomData<T>,
}

#[duplicate_item(
    f48       Wave48       Wave48Player;
    [ f64 ]   [ Wave64 ]   [ Wave64Player ];
    [ f32 ]   [ Wave32 ]   [ Wave32Player ];
)]
impl<T: Float> Wave48Player<T> {
    pub fn new(wave: Arc<Wave48>, channel: usize, loop_point: Option<usize>) -> Self {
        Self {
            wave,
            channel,
            index: 0,
            loop_point,
            _marker: PhantomData::default(),
        }
    }
}

#[duplicate_item(
    f48       Wave48       Wave48Player;
    [ f64 ]   [ Wave64 ]   [ Wave64Player ];
    [ f32 ]   [ Wave32 ]   [ Wave32Player ];
)]
impl<T: Float> AudioNode for Wave48Player<T> {
    const ID: u64 = 65;
    type Sample = T;
    type Inputs = typenum::U0;
    type Outputs = typenum::U1;

    fn reset(&mut self, _sample_rate: Option<f64>) {
        self.index = 0;
    }

    #[inline]
    fn tick(
        &mut self,
        _input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        if self.index < self.wave.length() {
            let value = self.wave.at(self.channel, self.index);
            self.index += 1;
            if self.index == self.wave.length() {
                if let Some(point) = self.loop_point {
                    self.index = point;
                }
            }
            [convert(value)].into()
        } else {
            [T::zero()].into()
        }
    }
}
