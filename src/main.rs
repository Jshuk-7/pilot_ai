mod rand {
    use std::time::{SystemTime, UNIX_EPOCH};

    extern "C" {
        fn srand(seed: u32) -> u32;
        fn rand() -> u32;
    }

    const RAND_MAX: u32 = 32_767;

    pub fn rand_f32() -> f32 {
        unsafe {
            srand(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u32,
            );
            //srand(69);

            rand() as f32 / RAND_MAX as f32
        }
    }
}

mod ai {
    const TRAIN: [[i32; 2]; 5] = [[0, 0], [1, 2], [2, 4], [3, 6], [4, 8]];

    pub fn cost(w: f32, b: f32) -> f32 {
        let mut result = 0.0;
        for i in 0..TRAIN.len() {
            let x = TRAIN[i][0] as f32;
            let y = x * w + b;
            let expected = TRAIN[i][1] as f32;
            let d = y - expected;
            result += d * d * d.abs();
        }

        result /= TRAIN.len() as f32;
        result
    }
}

// 1 000 000 000 000 => GPT-4
// 1 => us

// Model:
// 	Params: (w)
// 	y = xw;

fn main() {
    let mut w = rand::rand_f32() * 10.0;
    let mut b = rand::rand_f32() * 5.0;

    let eps = 1e-3 as f32;
    let rate = 1e-3;

    for _ in 0..500 {
        let cost = ai::cost(w, b);
        let dw = (ai::cost(w + eps, b) - cost) / eps;
        let db = (ai::cost(w, b + eps) - cost) / eps;
        w -= rate * dw;
        b -= rate * db;
        println!("cost: {:.05} w: {w:.05} b: {b:.05}", ai::cost(w, b));
    }

    println!("-----------------------------------------------");
    println!("{w} {b}");
}
