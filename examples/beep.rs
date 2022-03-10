#![allow(clippy::precedence)]

extern crate anyhow;
extern crate cpal;
extern crate fundsp;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use fundsp::hacker::*;

#[cfg_attr(target_os = "android", ndk_glue::main(backtrace = "full"))]
fn main() {
    // Conditionally compile with jack if the feature is specified.
    #[cfg(all(
        any(target_os = "linux", target_os = "dragonfly", target_os = "freebsd"),
        feature = "jack"
    ))]
    // Manually check for flags. Can be passed through cargo with -- e.g.
    // cargo run --release --example beep --features jack -- --jack
    let host = if std::env::args()
        .collect::<String>()
        .contains(&String::from("--jack"))
    {
        cpal::host_from_id(cpal::available_hosts()
            .into_iter()
            .find(|id| *id == cpal::HostId::Jack)
            .expect(
                "make sure --features jack is specified. only works on OSes where jack is available",
            )).expect("jack host unavailable")
    } else {
        cpal::default_host()
    };

    #[cfg(any(
        not(any(target_os = "linux", target_os = "dragonfly", target_os = "freebsd")),
        not(feature = "jack")
    ))]
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .expect("failed to find a default output device");
    let config = device.default_output_config().unwrap();

    match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into()).unwrap(),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into()).unwrap(),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into()).unwrap(),
    }
}

fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> Result<(), anyhow::Error>
where
    T: cpal::Sample,
{
    let sample_rate = config.sample_rate.0 as f64;
    let channels = config.channels as usize;

    //let c = mls();
    //let c = mls() >> lowpole_hz(400.0) >> lowpole_hz(400.0);
    //let c = (mls() | dc(500.0)) >> butterpass();
    //let c = (mls() | dc(400.0) | dc(50.0)) >> resonator();
    //let c = pink();

    //let f = 110.0;
    //let m = 5.0;
    //let c = sine_hz(f) * f * m + f >> sine();

    // Risset glissando.
    /*let c = stack::<U20, _, _>(|i| {
        lfo(move |t| {
            let f = lerp(-0.5, 0.5, rnd(i))
                + xerp(20.0, 20480.0, (t * 0.1 + i as f64 * 0.5) % 10.0 / 10.0);
            let a = smooth3(sin_hz(0.05, (t * 0.1 + i as f64 * 0.5) % 10.0));
            (a, f)
        }) >> pass() * sine() * 10.0
    }) >> multijoin::<U1, U20>()
        >> pinkpass();
    */

    // Pulse wave.
    let c = lfo(|t| {
        let pitch = 110.0;
        let duty = lerp11(0.01, 0.99, sin_hz(0.05, t));
        (pitch, duty)
    }) >> pulse();

    //let c = zero() >> pluck(220.0, 0.8, 0.8);
    //let c = dc(110.0) >> dsf_saw_r(0.99);
    //let c = dc(110.0) >> triangle();
    //let c = lfo(|t| xerp11(20.0, 2000.0, sin_hz(0.1, t))) >> dsf_square_r(0.99) >> lowpole_hz(1000.0);
    //let c = dc(110.0) >> square();

    // Test ease_noise.
    //let c = lfo(|t| xerp11(50.0, 5000.0, ease_noise(smooth9, 0, t))) >> triangle();

    //let c = c
    //    >> (pass() | envelope(|t| xerp(500.0, 20000.0, sin_hz(0.0666, t))) | dc(10.0))
    //    >> bandpass();

    //let c = c >> feedback(butterpass_hz(1000.0) >> delay(1.0) * 0.5);

    // Waveshapers.
    //let c = c >> shape_fn(|x| tanh(x * 5.0));

    let mut c = c
        >> declick() >> dcblock()
        >> split::<U2>()
        //>> reverb_stereo(0.2, 5.0)
        >> limiter_stereo((1.0, 5.0));
    //let mut c = c * 0.1;
    c.reset(Some(sample_rate));

    let mut next_value = move || c.get_stereo();

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            write_data(data, channels, &mut next_value)
        },
        err_fn,
    )?;
    stream.play()?;

    std::thread::sleep(std::time::Duration::from_millis(50000));

    Ok(())
}

fn write_data<T>(output: &mut [T], channels: usize, next_sample: &mut dyn FnMut() -> (f64, f64))
where
    T: cpal::Sample,
{
    for frame in output.chunks_mut(channels) {
        let sample = next_sample();
        let left: T = cpal::Sample::from::<f32>(&(sample.0 as f32));
        let right: T = cpal::Sample::from::<f32>(&(sample.1 as f32));

        for (channel, sample) in frame.iter_mut().enumerate() {
            if channel & 1 == 0 {
                *sample = left;
            } else {
                *sample = right;
            }
        }
    }
}
