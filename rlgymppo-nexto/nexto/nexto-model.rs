// Pre-generated Burn model and weights. Do not edit by hand.
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LinearLayout;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    layernormalization1: LayerNorm<B>,
    layernormalization2: LayerNorm<B>,
    linear5: Linear<B>,
    linear6: Linear<B>,
    linear7: Linear<B>,
    constant76: burn::module::Param<Tensor<B, 1, Int>>,
    constant77: burn::module::Param<Tensor<B, 1, Int>>,
    constant81: burn::module::Param<Tensor<B, 1>>,
    linear8: Linear<B>,
    constant89: burn::module::Param<Tensor<B, 1>>,
    layernormalization3: LayerNorm<B>,
    linear9: Linear<B>,
    linear10: Linear<B>,
    layernormalization4: LayerNorm<B>,
    layernormalization5: LayerNorm<B>,
    linear11: Linear<B>,
    linear12: Linear<B>,
    linear13: Linear<B>,
    constant111: burn::module::Param<Tensor<B, 1, Int>>,
    constant112: burn::module::Param<Tensor<B, 1, Int>>,
    constant116: burn::module::Param<Tensor<B, 1>>,
    linear14: Linear<B>,
    constant124: burn::module::Param<Tensor<B, 1>>,
    layernormalization6: LayerNorm<B>,
    linear15: Linear<B>,
    linear16: Linear<B>,
    linear17: Linear<B>,
    constant125: burn::module::Param<Tensor<B, 2>>,
    linear18: Linear<B>,
    linear19: Linear<B>,
    linear20: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}

#[repr(C, align(256))]
struct Aligned256([u8; 1786496usize]);
static ALIGNED_DATA: Aligned256 = Aligned256(*include_bytes!("nexto-model.bpk"));
static EMBEDDED_STATES: &[u8] = &ALIGNED_DATA.0;

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_embedded(&Default::default())
    }
}

impl<B: Backend> Model<B> {
    /// Load model weights from embedded burnpack data (zero-copy at store level).
    ///
    /// The embedded data stays in the binary's .rodata section without heap allocation.
    /// Tensor data is sliced directly from the static bytes.
    ///
    /// Note: Some backends may still copy data internally.
    /// See <https://github.com/tracel-ai/burn/issues/4153> for true backend zero-copy.
    ///
    /// See <https://github.com/tracel-ai/burn/issues/4123>
    pub fn from_embedded(device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_static(EMBEDDED_STATES);
        model
            .load_from(&mut store)
            .expect("Failed to load embedded burnpack");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(32, 128).with_bias(true).init(device);
        let linear2 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear3 = LinearConfig::new(24, 128).with_bias(true).init(device);
        let linear4 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let layernormalization1 = LayerNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let layernormalization2 = LayerNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear5 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear6 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear7 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let constant76: burn::module::Param<Tensor<B, 1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1, Int>::zeros([4], (device, burn::tensor::DType::I64))
            },
            device.clone(),
            false,
            [4].into(),
        );
        let constant77: burn::module::Param<Tensor<B, 1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1, Int>::zeros([4], (device, burn::tensor::DType::I64))
            },
            device.clone(),
            false,
            [4].into(),
        );
        let constant81: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([5.656854152679443f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let linear8 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let constant89: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([4f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let layernormalization3 = LayerNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear9 = LinearConfig::new(128, 512).with_bias(true).init(device);
        let linear10 = LinearConfig::new(512, 128).with_bias(true).init(device);
        let layernormalization4 = LayerNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let layernormalization5 = LayerNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear11 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear12 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear13 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let constant111: burn::module::Param<Tensor<B, 1, Int>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1, Int>::zeros([4], (device, burn::tensor::DType::I64))
                },
                device.clone(),
                false,
                [4].into(),
            );
        let constant112: burn::module::Param<Tensor<B, 1, Int>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1, Int>::zeros([4], (device, burn::tensor::DType::I64))
                },
                device.clone(),
                false,
                [4].into(),
            );
        let constant116: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([5.656854152679443f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let linear14 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let constant124: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([4f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let layernormalization6 = LayerNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear15 = LinearConfig::new(128, 512).with_bias(true).init(device);
        let linear16 = LinearConfig::new(512, 128).with_bias(true).init(device);
        let linear17 = LinearConfig::new(128, 32).with_bias(true).init(device);
        let constant125: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 2>::zeros([90, 8], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [90, 8].into(),
        );
        let linear18 = LinearConfig::new(8, 32)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear19 = LinearConfig::new(32, 32)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear20 = LinearConfig::new(32, 32)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            linear1,
            linear2,
            linear3,
            linear4,
            layernormalization1,
            layernormalization2,
            linear5,
            linear6,
            linear7,
            constant76,
            constant77,
            constant81,
            linear8,
            constant89,
            layernormalization3,
            linear9,
            linear10,
            layernormalization4,
            layernormalization5,
            linear11,
            linear12,
            linear13,
            constant111,
            constant112,
            constant116,
            linear14,
            constant124,
            layernormalization6,
            linear15,
            linear16,
            linear17,
            constant125,
            linear18,
            linear19,
            linear20,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        q: Tensor<B, 3>,
        kv: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>, Tensor<B, 3>) {
        let linear1_out1 = self.linear1.forward(q);
        let relu1_out1 = burn::tensor::activation::relu(linear1_out1);
        let linear2_out1 = self.linear2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(linear2_out1);
        let linear3_out1 = self.linear3.forward(kv);
        let relu3_out1 = burn::tensor::activation::relu(linear3_out1);
        let linear4_out1 = self.linear4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(linear4_out1);
        let layernormalization1_out1 = {
            let dtype = relu2_out1.clone().dtype();
            self.layernormalization1
                .forward(relu2_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let layernormalization2_out1 = {
            let dtype = relu4_out1.clone().dtype();
            self.layernormalization2
                .forward(relu4_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose1_out1 = layernormalization1_out1.permute([1, 0, 2]);
        let transpose2_out1 = layernormalization2_out1.permute([1, 0, 2]);
        let shape1_out1: [i64; 3] = {
            let axes = &transpose1_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather1_out1 = shape1_out1[0] as i64;
        let gather2_out1 = shape1_out1[1] as i64;
        let gather3_out1 = shape1_out1[2] as i64;
        let constant58_out1 = 4i64;
        let div1_out1 = gather3_out1 / constant58_out1;
        let linear5_out1 = self.linear5.forward(transpose1_out1);
        let linear6_out1 = self.linear6.forward(transpose2_out1.clone());
        let linear7_out1 = self.linear7.forward(transpose2_out1);
        let constant59_out1 = 4i64;
        let mul1_out1 = gather2_out1 * constant59_out1;
        let unsqueeze1_out1 = [gather1_out1 as i64];
        let unsqueeze2_out1 = [mul1_out1 as i64];
        let unsqueeze3_out1 = [div1_out1 as i64];
        let concat1_out1: [i64; 3usize] = [
            &unsqueeze1_out1[..],
            &unsqueeze2_out1[..],
            &unsqueeze3_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape1_out1 = linear5_out1.reshape(concat1_out1);
        let transpose3_out1 = reshape1_out1.permute([1, 0, 2]);
        let constant63_out1: [i64; 1] = [-1i64];
        let unsqueeze4_out1 = [mul1_out1 as i64];
        let unsqueeze5_out1 = [div1_out1 as i64];
        let concat2_out1: [i64; 3usize] = [
            &constant63_out1[..],
            &unsqueeze4_out1[..],
            &unsqueeze5_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let constant66_out1: [i64; 1] = [-1i64];
        let unsqueeze6_out1 = [mul1_out1 as i64];
        let unsqueeze7_out1 = [div1_out1 as i64];
        let concat3_out1: [i64; 3usize] = [
            &constant66_out1[..],
            &unsqueeze6_out1[..],
            &unsqueeze7_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape2_out1 = linear6_out1.reshape(concat2_out1);
        let transpose4_out1 = reshape2_out1.clone().permute([1, 0, 2]);
        let reshape3_out1 = linear7_out1.reshape(concat3_out1);
        let transpose5_out1 = reshape3_out1.permute([1, 0, 2]);
        let shape4_out1: [i64; 3] = {
            let axes = &transpose4_out1.dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather4_out1 = shape4_out1[1] as i64;
        let unsqueeze8_out1 = [gather2_out1 as i64];
        let constant71_out1: [i64; 1] = [1i64];
        let constant72_out1: [i64; 1] = [1i64];
        let unsqueeze9_out1 = [gather4_out1 as i64];
        let concat4_out1: [i64; 4usize] = [
            &unsqueeze8_out1[..],
            &constant71_out1[..],
            &constant72_out1[..],
            &unsqueeze9_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape4_out1 = mask.clone().reshape(concat4_out1);
        let constantofshape1_out1: [i64; 4usize] = [1i64, 1i64, 1i64, 1i64];
        let constant75_out1 = -1i64;
        let mul2_out1 = {
            let mut result = constantofshape1_out1;
            let __scalar = constant75_out1 as i64;
            for result_item in result.iter_mut() {
                *result_item = result_item.saturating_mul(__scalar);
            }
            result
        };
        let constant76_out1 = self.constant76.val();
        let equal1_out1 = {
            let shape_tensor = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::from(mul2_out1.as_slice()),
                (&self.device, burn::tensor::DType::I64),
            );
            constant76_out1.equal(shape_tensor)
        };
        let constant77_out1 = self.constant77.val();
        let where1_out1 = constant77_out1.mask_where(
            equal1_out1,
            Tensor::<B, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::from(&constantofshape1_out1 as &[i64]),
                (&self.device, burn::tensor::DType::I64),
            ),
        );
        let expand1_out1 = {
            let onnx_shape: [i64; 4usize] = TryInto::<[i64; 4usize]>::try_into(
                where1_out1.to_data().convert::<i64>().as_slice().unwrap(),
            )
            .unwrap();
            let input_dims = reshape4_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            reshape4_out1.expand(shape)
        };
        let unsqueeze10_out1 = [mul1_out1 as i64];
        let constant79_out1: [i64; 1] = [1i64];
        let unsqueeze11_out1 = [gather4_out1 as i64];
        let concat5_out1: [i64; 3usize] = [
            &unsqueeze10_out1[..],
            &constant79_out1[..],
            &unsqueeze11_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape5_out1 = expand1_out1.reshape(concat5_out1);
        let constant81_out1 = self.constant81.val();
        let div2_out1 = transpose3_out1.div((constant81_out1).unsqueeze_dims(&[0isize, 1isize]));
        let transpose6_out1 = reshape2_out1.permute([1, 2, 0]);
        let matmul8_out1 = div2_out1.matmul(transpose6_out1);
        let add1_out1 = matmul8_out1.add(reshape5_out1);
        let softmax1_out1 = burn::tensor::activation::softmax(add1_out1, 2);
        let matmul9_out1 = softmax1_out1.clone().matmul(transpose5_out1);
        let transpose7_out1 = matmul9_out1.permute([1, 0, 2]);
        let unsqueeze12_out1 = [gather1_out1 as i64];
        let unsqueeze13_out1 = [gather2_out1 as i64];
        let unsqueeze14_out1 = [gather3_out1 as i64];
        let concat6_out1: [i64; 3usize] = [
            &unsqueeze12_out1[..],
            &unsqueeze13_out1[..],
            &unsqueeze14_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape6_out1 = transpose7_out1.reshape(concat6_out1);
        let linear8_out1 = self.linear8.forward(reshape6_out1);
        let unsqueeze15_out1 = [gather2_out1 as i64];
        let constant86_out1: [i64; 1] = [4i64];
        let unsqueeze16_out1 = [gather1_out1 as i64];
        let unsqueeze17_out1 = [gather4_out1 as i64];
        let concat7_out1: [i64; 4usize] = [
            &unsqueeze15_out1[..],
            &constant86_out1[..],
            &unsqueeze16_out1[..],
            &unsqueeze17_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape7_out1 = softmax1_out1.reshape(concat7_out1);
        let reducesum1_out1 = { reshape7_out1.sum_dim(1usize).squeeze_dims::<3usize>(&[1]) };
        let constant89_out1 = self.constant89.val();
        let div3_out1 = reducesum1_out1.div((constant89_out1).unsqueeze_dims(&[0isize, 1isize]));
        let transpose8_out1 = linear8_out1.permute([1, 0, 2]);
        let add2_out1 = relu2_out1.add(transpose8_out1);
        let layernormalization3_out1 = {
            let dtype = add2_out1.clone().dtype();
            self.layernormalization3
                .forward(add2_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear9_out1 = self.linear9.forward(layernormalization3_out1);
        let relu5_out1 = burn::tensor::activation::relu(linear9_out1);
        let linear10_out1 = self.linear10.forward(relu5_out1);
        let add3_out1 = add2_out1.add(linear10_out1);
        let layernormalization4_out1 = {
            let dtype = add3_out1.clone().dtype();
            self.layernormalization4
                .forward(add3_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let layernormalization5_out1 = {
            let dtype = relu4_out1.dtype();
            self.layernormalization5
                .forward(relu4_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose9_out1 = layernormalization4_out1.permute([1, 0, 2]);
        let transpose10_out1 = layernormalization5_out1.permute([1, 0, 2]);
        let shape5_out1: [i64; 3] = {
            let axes = &transpose9_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather5_out1 = shape5_out1[0] as i64;
        let gather6_out1 = shape5_out1[1] as i64;
        let gather7_out1 = shape5_out1[2] as i64;
        let constant93_out1 = 4i64;
        let div4_out1 = gather7_out1 / constant93_out1;
        let linear11_out1 = self.linear11.forward(transpose9_out1);
        let linear12_out1 = self.linear12.forward(transpose10_out1.clone());
        let linear13_out1 = self.linear13.forward(transpose10_out1);
        let constant94_out1 = 4i64;
        let mul3_out1 = gather6_out1 * constant94_out1;
        let unsqueeze18_out1 = [gather5_out1 as i64];
        let unsqueeze19_out1 = [mul3_out1 as i64];
        let unsqueeze20_out1 = [div4_out1 as i64];
        let concat8_out1: [i64; 3usize] = [
            &unsqueeze18_out1[..],
            &unsqueeze19_out1[..],
            &unsqueeze20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape8_out1 = linear11_out1.reshape(concat8_out1);
        let transpose11_out1 = reshape8_out1.permute([1, 0, 2]);
        let constant98_out1: [i64; 1] = [-1i64];
        let unsqueeze21_out1 = [mul3_out1 as i64];
        let unsqueeze22_out1 = [div4_out1 as i64];
        let concat9_out1: [i64; 3usize] = [
            &constant98_out1[..],
            &unsqueeze21_out1[..],
            &unsqueeze22_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let constant101_out1: [i64; 1] = [-1i64];
        let unsqueeze23_out1 = [mul3_out1 as i64];
        let unsqueeze24_out1 = [div4_out1 as i64];
        let concat10_out1: [i64; 3usize] = [
            &constant101_out1[..],
            &unsqueeze23_out1[..],
            &unsqueeze24_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape9_out1 = linear12_out1.reshape(concat9_out1);
        let transpose12_out1 = reshape9_out1.clone().permute([1, 0, 2]);
        let reshape10_out1 = linear13_out1.reshape(concat10_out1);
        let transpose13_out1 = reshape10_out1.permute([1, 0, 2]);
        let shape8_out1: [i64; 3] = {
            let axes = &transpose12_out1.dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather8_out1 = shape8_out1[1] as i64;
        let unsqueeze25_out1 = [gather6_out1 as i64];
        let constant106_out1: [i64; 1] = [1i64];
        let constant107_out1: [i64; 1] = [1i64];
        let unsqueeze26_out1 = [gather8_out1 as i64];
        let concat11_out1: [i64; 4usize] = [
            &unsqueeze25_out1[..],
            &constant106_out1[..],
            &constant107_out1[..],
            &unsqueeze26_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape11_out1 = mask.reshape(concat11_out1);
        let constantofshape2_out1: [i64; 4usize] = [1i64, 1i64, 1i64, 1i64];
        let constant110_out1 = -1i64;
        let mul4_out1 = {
            let mut result = constantofshape2_out1;
            let __scalar = constant110_out1 as i64;
            for result_item in result.iter_mut() {
                *result_item = result_item.saturating_mul(__scalar);
            }
            result
        };
        let constant111_out1 = self.constant111.val();
        let equal2_out1 = {
            let shape_tensor = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::from(mul4_out1.as_slice()),
                (&self.device, burn::tensor::DType::I64),
            );
            constant111_out1.equal(shape_tensor)
        };
        let constant112_out1 = self.constant112.val();
        let where2_out1 = constant112_out1.mask_where(
            equal2_out1,
            Tensor::<B, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::from(&constantofshape2_out1 as &[i64]),
                (&self.device, burn::tensor::DType::I64),
            ),
        );
        let expand2_out1 = {
            let onnx_shape: [i64; 4usize] = TryInto::<[i64; 4usize]>::try_into(
                where2_out1.to_data().convert::<i64>().as_slice().unwrap(),
            )
            .unwrap();
            let input_dims = reshape11_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            reshape11_out1.expand(shape)
        };
        let unsqueeze27_out1 = [mul3_out1 as i64];
        let constant114_out1: [i64; 1] = [1i64];
        let unsqueeze28_out1 = [gather8_out1 as i64];
        let concat12_out1: [i64; 3usize] = [
            &unsqueeze27_out1[..],
            &constant114_out1[..],
            &unsqueeze28_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape12_out1 = expand2_out1.reshape(concat12_out1);
        let constant116_out1 = self.constant116.val();
        let div5_out1 = transpose11_out1.div((constant116_out1).unsqueeze_dims(&[0isize, 1isize]));
        let transpose14_out1 = reshape9_out1.permute([1, 2, 0]);
        let matmul16_out1 = div5_out1.matmul(transpose14_out1);
        let add4_out1 = matmul16_out1.add(reshape12_out1);
        let softmax2_out1 = burn::tensor::activation::softmax(add4_out1, 2);
        let matmul17_out1 = softmax2_out1.clone().matmul(transpose13_out1);
        let transpose15_out1 = matmul17_out1.permute([1, 0, 2]);
        let unsqueeze29_out1 = [gather5_out1 as i64];
        let unsqueeze30_out1 = [gather6_out1 as i64];
        let unsqueeze31_out1 = [gather7_out1 as i64];
        let concat13_out1: [i64; 3usize] = [
            &unsqueeze29_out1[..],
            &unsqueeze30_out1[..],
            &unsqueeze31_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape13_out1 = transpose15_out1.reshape(concat13_out1);
        let linear14_out1 = self.linear14.forward(reshape13_out1);
        let unsqueeze32_out1 = [gather6_out1 as i64];
        let constant121_out1: [i64; 1] = [4i64];
        let unsqueeze33_out1 = [gather5_out1 as i64];
        let unsqueeze34_out1 = [gather8_out1 as i64];
        let concat14_out1: [i64; 4usize] = [
            &unsqueeze32_out1[..],
            &constant121_out1[..],
            &unsqueeze33_out1[..],
            &unsqueeze34_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape14_out1 = softmax2_out1.reshape(concat14_out1);
        let reducesum2_out1 = { reshape14_out1.sum_dim(1usize).squeeze_dims::<3usize>(&[1]) };
        let constant124_out1 = self.constant124.val();
        let div6_out1 = reducesum2_out1.div((constant124_out1).unsqueeze_dims(&[0isize, 1isize]));
        let transpose16_out1 = linear14_out1.permute([1, 0, 2]);
        let add5_out1 = add3_out1.add(transpose16_out1);
        let layernormalization6_out1 = {
            let dtype = add5_out1.clone().dtype();
            self.layernormalization6
                .forward(add5_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear15_out1 = self.linear15.forward(layernormalization6_out1);
        let relu6_out1 = burn::tensor::activation::relu(linear15_out1);
        let linear16_out1 = self.linear16.forward(relu6_out1);
        let add6_out1 = add5_out1.add(linear16_out1);
        let relu7_out1 = burn::tensor::activation::relu(add6_out1);
        let linear17_out1 = self.linear17.forward(relu7_out1);
        let constant125_out1 = self.constant125.val();
        let linear18_out1 = self.linear18.forward(constant125_out1);
        let relu8_out1 = burn::tensor::activation::relu(linear18_out1);
        let linear19_out1 = self.linear19.forward(relu8_out1);
        let relu9_out1 = burn::tensor::activation::relu(linear19_out1);
        let linear20_out1 = self.linear20.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(linear20_out1);
        let einsum1_out1 = {
            let einsum_lhs = relu10_out1;
            let einsum_rhs = linear17_out1.permute([2usize, 0usize, 1usize]);
            let einsum_lhs_shape = einsum_lhs.dims();
            let einsum_rhs_shape = einsum_rhs.dims();
            let einsum_lhs_3d: Tensor<B, 3> =
                einsum_lhs.reshape([1usize, einsum_lhs_shape[0usize], einsum_lhs_shape[1usize]]);
            let einsum_rhs_3d: Tensor<B, 3> = einsum_rhs.reshape([
                1usize,
                einsum_lhs_shape[1usize],
                einsum_rhs_shape[1usize] * einsum_rhs_shape[2usize],
            ]);
            let einsum_result: Tensor<B, 3> = einsum_lhs_3d.matmul(einsum_rhs_3d).reshape([
                einsum_lhs_shape[0usize],
                einsum_rhs_shape[1usize],
                einsum_rhs_shape[2usize],
            ]);
            einsum_result.permute([1usize, 2usize, 0usize])
        };
        let gather9_out1 = {
            let sliced = einsum1_out1.slice(s![.., 0, ..]);
            sliced.squeeze_dim::<2usize>(1)
        };
        (gather9_out1, div3_out1, div6_out1)
    }
}
