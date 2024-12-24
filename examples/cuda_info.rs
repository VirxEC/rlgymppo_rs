use tch::Cuda;

fn main() {
    println!("Cuda is available: {}", Cuda::is_available());
    println!("Number of CUDA devices: {}", Cuda::device_count());
}
