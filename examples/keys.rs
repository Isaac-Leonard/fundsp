//! Play notes interactively on a virtual keyboard.
//! Please run me in release mode!
#![allow(clippy::precedence)]

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use eframe::egui;
use egui::*;
use fundsp::hacker::*;
use funutd::Rnd;

#[derive(Debug, PartialEq)]
enum Waveform {
    Sine,
    Saw,
    Square,
    Triangle,
    Organ,
    Hammond,
    Pulse,
    Pluck,
    Noise,
}

#[derive(Debug, PartialEq)]
enum Filter {
    None,
    Moog,
    Butterworth,
    Bandpass,
    Peak,
}

#[allow(dead_code)]
struct State {
    /// Random number generator.
    rnd: Rnd,
    /// Status of keys.
    id: Vec<Option<EventId>>,
    /// Sequencer frontend.
    sequencer: Sequencer64,
    /// Network frontend.
    net: Net64,
    /// Selected waveform.
    waveform: Waveform,
    /// Selected filter.
    filter: Filter,
    /// Vibrato amount in 0...1.
    vibrato_amount: f64,
    /// Chorus amount.
    chorus_amount: Shared<f64>,
    /// Reverb amount.
    reverb_amount: Shared<f64>,
    /// Reverb room size.
    room_size: f64,
    /// Reverb time in seconds.
    reverb_time: f64,
    /// Reverb frontend.
    reverb: Slot64,
    /// Left channel data for the oscilloscope.
    snoop0: Snoop<f64>,
    /// Right channel data for the oscilloscope.
    snoop1: Snoop<f64>,
}

static KEYS: [Key; 29] = [
    Key::Z,
    Key::S,
    Key::X,
    Key::D,
    Key::C,
    Key::V,
    Key::G,
    Key::B,
    Key::H,
    Key::N,
    Key::J,
    Key::M,
    Key::Q,
    Key::Num2,
    Key::W,
    Key::Num3,
    Key::E,
    Key::R,
    Key::Num5,
    Key::T,
    Key::Num6,
    Key::Y,
    Key::Num7,
    Key::U,
    Key::I,
    Key::Num9,
    Key::O,
    Key::Num0,
    Key::P,
];

fn main() {
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .expect("failed to find a default output device");
    let config = device.default_output_config().unwrap();

    match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into()).unwrap(),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into()).unwrap(),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into()).unwrap(),
        _ => panic!("Unsupported format"),
    }
}

fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> Result<(), anyhow::Error>
where
    T: SizedSample + FromSample<f64>,
{
    let sample_rate = config.sample_rate.0 as f64;
    let channels = config.channels as usize;

    let mut sequencer = Sequencer64::new(false, 1);
    let sequencer_backend = sequencer.backend();

    let (snoop0, snoop_backend0) = snoop(32768);
    let (snoop1, snoop_backend1) = snoop(32768);

    let room_size = 10.0;
    let reverb_amount = shared(0.2);
    let reverb_time = 2.0;
    let chorus_amount = shared(1.0);

    let mut net = Net64::wrap(Box::new(sequencer_backend));
    let (reverb, reverb_backend) =
        Slot64::new(Box::new(reverb2_stereo(room_size, reverb_time, 0.5)));
    net = net >> pan(0.0);
    // Smooth chorus and reverb amounts to prevent discontinuities.
    net = net
        >> ((1.0 - var(&chorus_amount) >> follow(0.01) >> split()) * multipass()
            & (var(&chorus_amount) >> follow(0.01) >> split())
                * (chorus(0, 0.0, 0.02, 0.3) | chorus(1, 0.0, 0.02, 0.3)));
    net = net
        >> ((1.0 - var(&reverb_amount) >> follow(0.01) >> split::<U2>()) * multipass()
            & (var(&reverb_amount) >> follow(0.01) >> split::<U2>())
                * Net64::wrap(Box::new(reverb_backend)))
        >> (snoop_backend0 | snoop_backend1);

    net.set_sample_rate(sample_rate);

    // Use block processing for maximum efficiency.
    let mut backend = BlockRateAdapter64::new(Box::new(net.backend()));

    let mut next_value = move || backend.get_stereo();

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            write_data(data, channels, &mut next_value)
        },
        err_fn,
        None,
    )?;
    stream.play()?;

    let viewport = ViewportBuilder::default().with_min_inner_size(vec2(360.0, 480.0));

    let options = eframe::NativeOptions {
        viewport,
        ..eframe::NativeOptions::default()
    };

    let state: State = State {
        rnd: Rnd::from_time(),
        id: vec![None; KEYS.len()],
        sequencer,
        net,
        waveform: Waveform::Saw,
        filter: Filter::None,
        vibrato_amount: 0.5,
        chorus_amount,
        reverb_amount,
        room_size,
        reverb,
        reverb_time,
        snoop0,
        snoop1,
    };

    eframe::run_native(
        "Virtual Keyboard Example",
        options,
        Box::new(|_cc| Box::new(state)),
    )
    .unwrap();

    Ok(())
}

fn write_data<T>(output: &mut [T], channels: usize, next_sample: &mut dyn FnMut() -> (f64, f64))
where
    T: SizedSample + FromSample<f64>,
{
    for frame in output.chunks_mut(channels) {
        let sample = next_sample();
        let left: T = T::from_sample(sample.0);
        let right: T = T::from_sample(sample.1);

        for (channel, sample) in frame.iter_mut().enumerate() {
            if channel & 1 == 0 {
                *sample = left;
            } else {
                *sample = right;
            }
        }
    }
}

impl eframe::App for State {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::dark());
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Virtual Keyboard Example");
            ui.separator();
            ui.end_row();

            ui.label("Waveform");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.waveform, Waveform::Sine, "Sine");
                ui.selectable_value(&mut self.waveform, Waveform::Saw, "Saw");
                ui.selectable_value(&mut self.waveform, Waveform::Square, "Square");
                ui.selectable_value(&mut self.waveform, Waveform::Triangle, "Triangle");
                ui.selectable_value(&mut self.waveform, Waveform::Organ, "Organ");
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.waveform, Waveform::Hammond, "Hammond");
                ui.selectable_value(&mut self.waveform, Waveform::Pulse, "Pulse");
                ui.selectable_value(&mut self.waveform, Waveform::Pluck, "Pluck");
                ui.selectable_value(&mut self.waveform, Waveform::Noise, "Noise");
            });
            ui.separator();
            ui.end_row();

            ui.label("Vibrato Amount");
            let mut vibrato = self.vibrato_amount * 100.0;
            ui.add(egui::Slider::new(&mut vibrato, 0.0..=100.0).suffix("%"));
            self.vibrato_amount = vibrato * 0.01;
            ui.separator();
            ui.end_row();

            ui.label("Filter");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.filter, Filter::None, "None");
                ui.selectable_value(&mut self.filter, Filter::Moog, "Moog");
                ui.selectable_value(&mut self.filter, Filter::Butterworth, "Butterworth");
                ui.selectable_value(&mut self.filter, Filter::Bandpass, "Bandpass");
                ui.selectable_value(&mut self.filter, Filter::Peak, "Peak");
            });
            ui.separator();
            ui.end_row();

            ui.label("Chorus Amount");
            let mut chorus = self.chorus_amount.value() * 100.0;
            ui.add(egui::Slider::new(&mut chorus, 0.0..=100.0).suffix("%"));
            self.chorus_amount.set_value(chorus * 0.01);
            ui.separator();
            ui.end_row();

            ui.label("Reverb Amount");
            let mut reverb = self.reverb_amount.value() * 100.0;
            ui.add(egui::Slider::new(&mut reverb, 0.0..=100.0).suffix("%"));
            self.reverb_amount.set_value(reverb * 0.01);
            let mut reverb_time = self.reverb_time;
            let mut room_size = self.room_size;
            ui.label("Reverb Time");
            ui.add(egui::Slider::new(&mut reverb_time, 1.0..=10.0).suffix("s"));
            ui.label("Reverb Room Size");
            ui.add(egui::Slider::new(&mut room_size, 10.0..=30.0).suffix("m"));
            if self.room_size != room_size || self.reverb_time != reverb_time {
                self.reverb.set(
                    Fade::Smooth,
                    0.5,
                    Box::new(reverb2_stereo(room_size, reverb_time, 0.5)),
                );
                self.room_size = room_size;
                self.reverb_time = reverb_time;
            }
            ui.separator();
            ui.end_row();

            // Draw oscilloscope.
            egui::containers::Frame::canvas(ui.style()).show(ui, |ui| {
                ui.ctx().request_repaint();

                self.snoop0.update();
                self.snoop1.update();

                let points = 512;
                let color0 = Color32::from_rgb(180, 200, 220);
                let color1 = Color32::from_rgb(200, 200, 200);
                let thickness: f32 = 1.0;

                let desired_size = ui.available_width() * vec2(1.0, 0.25);
                let (_id, rect) = ui.allocate_space(desired_size);

                let to_screen = emath::RectTransform::from_to(
                    Rect::from_x_y_ranges(0.0..=points as f32, -1.0..=1.0),
                    rect,
                );

                let points0: Vec<Pos2> = (0..points)
                    .map(|i| {
                        let y = self.snoop0.at(i);
                        to_screen * pos2((points - i) as f32, softsign(y * 10.0) as f32)
                    })
                    .collect();
                let line0 = epaint::Shape::line(points0, Stroke::new(thickness, color0));
                let points1: Vec<Pos2> = (0..points)
                    .map(|i| {
                        let y = self.snoop1.at(i);
                        to_screen * pos2((points - i) as f32, softsign(y * 10.0) as f32)
                    })
                    .collect();
                let line1 = epaint::Shape::line(points1, Stroke::new(thickness, color1));
                ui.painter().add(line0);
                ui.painter().add(line1);
            });

            #[allow(clippy::needless_range_loop)]
            for i in 0..KEYS.len() {
                if ctx.input(|c| !c.key_down(KEYS[i])) {
                    if let Some(id) = self.id[i] {
                        // Start fading out existing note.
                        self.sequencer.edit_relative(id, 0.2, 0.2);
                        self.id[i] = None;
                    }
                }
                if ctx.input(|c| c.key_down(KEYS[i])) && self.id[i].is_none() {
                    let pitch_hz = midi_hz(40.0 + i as f64);
                    let v = self.vibrato_amount * 0.003;
                    let pitch = lfo(move |t| {
                        pitch_hz * xerp11(1.0 / (1.0 + v), 1.0 + v, sin_hz(6.0, t) + sin_hz(6.1, t))
                    });
                    let waveform = match self.waveform {
                        Waveform::Sine => Net64::wrap(Box::new(pitch * 2.0 >> sine() * 0.1)),
                        Waveform::Saw => Net64::wrap(Box::new(pitch >> saw() * 0.5)),
                        Waveform::Square => Net64::wrap(Box::new(pitch >> square() * 0.5)),
                        Waveform::Triangle => Net64::wrap(Box::new(pitch >> triangle() * 0.5)),
                        Waveform::Organ => Net64::wrap(Box::new(pitch >> organ() * 0.5)),
                        Waveform::Hammond => Net64::wrap(Box::new(pitch >> hammond() * 0.5)),
                        Waveform::Pulse => Net64::wrap(Box::new(
                            (pitch | lfo(move |t| lerp11(0.01, 0.99, sin_hz(0.1, t))))
                                >> pulse() * 0.5,
                        )),
                        Waveform::Pluck => {
                            Net64::wrap(Box::new(zero() >> pluck(pitch_hz, 0.5, 0.5) * 0.5))
                        }
                        Waveform::Noise => Net64::wrap(Box::new(
                            (noise()
                                | pitch * 4.0
                                | lfo(move |t| funutd::math::lerp(100.0, 20.0, clamp01(t * 5.0))))
                                >> !resonator()
                                >> resonator()
                                >> shape(fundsp::shape::Shape::AdaptiveTanh(0.01, 0.1)),
                        )),
                    };
                    let filter = match self.filter {
                        Filter::None => Net64::wrap(Box::new(pass())),
                        Filter::Moog => Net64::wrap(Box::new(
                            (pass() | lfo(move |t| (max(200.0, 10000.0 * exp(-t)), 0.6))) >> moog(),
                        )),
                        Filter::Butterworth => Net64::wrap(Box::new(
                            (pass() | lfo(move |t| max(200.0, 10000.0 * exp(-t * 5.0))))
                                >> butterpass(),
                        )),
                        Filter::Bandpass => Net64::wrap(Box::new(
                            (pass() | lfo(move |t| (xerp11(200.0, 10000.0, sin_hz(0.2, t)), 2.0)))
                                >> bandpass(),
                        )),
                        Filter::Peak => Net64::wrap(Box::new(
                            (pass() | lfo(move |t| (xerp11(200.0, 10000.0, sin_hz(0.2, t)), 2.0)))
                                >> peak(),
                        )),
                    };
                    let mut note = Box::new(waveform >> filter);
                    // Give the note its own random seed.
                    note.ping(false, AttoHash::new(self.rnd.u64()));
                    // Insert new note. We set the end time to infinity initially,
                    // which means it plays indefinitely until the key is released.
                    self.id[i] = Some(self.sequencer.push_relative(
                        0.0,
                        f64::INFINITY,
                        Fade::Smooth,
                        0.02,
                        0.2,
                        note,
                    ));
                }
            }
        });
    }
}
