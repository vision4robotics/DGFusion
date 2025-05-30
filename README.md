# DGFusion: Discrepancy-Guided Network for Aerial Visible-Thermal Image Fusion

Changhong Fu, Xiang Huang, Zijie Zhang, Haobo Zuo, Yongkang Cao, Shenghao Ren
<img src="./asserts/framework.svg" width="800px"/>


## <a name="abstract"></a>📎 Abstract
>Visible-thermal image fusion is vital for unmanned aerial vehicle to achieve intelligent applications. However, existing fusion methods mainly rely on extracted inherent modality features without explicit attention to cross-modality discrepancies and effective feature enhancement mechanisms, critically limiting fusion performance. To address these issues, a novel discrepancy-guided visible-thermal image fusion (DGFusion) is introduced for UAV. Specifically, a dedicated amplitude-phase feature enhancer is introduced, augmenting inherent modality features in order to reinforce global information while preserving local clarity. Additionally, an innovative discrepancy-aware feature extractor is presented to highlight critical modality-discrepancy cues while filtering out irrelevant noise. Besides, an adaptive discrepancy-guided feature fusion is proposed to integrate both enhanced and discrepant features, enabling stable fusion results under dynamic UAV conditions. A new dataset comprising 8700 pairs of visible-thermal images captured from a UAV perspective and covering diverse challenging scenarios is established to evaluate the effectiveness of the proposed fusion method. Extensive experiments demonstrate that DGFusion outperforms state-of-the-art methods in terms of both fusion quality and robust performance.

## <a name="Benchmark"></a> 🌈 Benchmark
The benchmark encompasses 8,700 aligned visible-thermal image pairs, of which 6,393 pairs are designated for training and 2,307 pairs for testing.You can download the benchmark from [DGFusion_Benchmark]()
<img src="./asserts/dataset.svg" width="800px"/>

## 🚀 : How to train

1. Run `VSM.m` to generate saliency maps for dataset.

2. Update the paths of image sets in the training configuration.

    ```bash
    Updating the paths in configure files of /options/train/DGFusion.yml
    ```

3. Run the training commands.

    ```bash
    python train.py -opt /SHIP/options/train/DGFusion.yml
    ```

---

## 🚀 : How to test

1. Download the pretrained model [DGFusion]() and put it in the `pretrained_model` folder.

2. Update the paths of image sets and pretrained models.

    ```bash
    Updating the paths in configure files of /options/test/DGFusion.yml
    ```

3. Run the testing commands.

    ```bash
    python test.py -opt /options/test/DGFusion.yml
    ```



