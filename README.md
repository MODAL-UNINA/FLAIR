## Abstract
Deep learning (DL) has significantly advanced Industry 
4.0 by leveraging data from the Industrial Internet of Things (IIoT) to enable smart manufacturing, predictive maintenance, and data-driven product marketing. However, multimodal industrial data presents challenges for traditional frameworks, including scalability, data privacy, and integration efficiency. This paper introduces an efficient product retrieval framework for e-commerce systems, addressing privacy and performance challenges through federated learning (FL). Specifically, we propose \textcolor{blue}{FLAIR (Federated Learning for Augmented Industrial Retrieval)}, a novel part retrieval system where distributed warehouses collaboratively train a multimodal foundation model, CLIP (Contrastive Language-Image Pre-Training), by fine-tuning only the Adapter module via FL, ensuring data privacy and efficiency. To address the limited availability of multimodal industrial data, our framework incorporates effective data augmentation strategies to enhance the diversity and quality of the training dataset. Comprehensive experiments on the Industrial Language-Image Dataset (ILID) highlight that FLAIR holds effective privacy safeguards and strong retrieval capabilities. Additionally, an advanced e-commerce recommendation system built on FLAIR showcases its practical effectiveness. FLAIR represents the first application of FL for industrial product retrieval, optimizing part searches, inventory management, and customer experience while maintaining data security.


### Note
1. Download ILID data

   please download Industrial Language-Image Dataset (ILID) from [here](https://github.com/kenomo/industrial-clip)

2. Example data folder structure
   
```
├── data
│   ├── images
│   │   ├── accordion_roller_conveyor
│   │   │   ├─ 485bfc5b-48bb-47f6-b4f2-6f8c6412590b.png
│   │   │   └── ...
│   │   ├── accumulating_pallet_stopping
│   │   │   ├── a245f52c-5caf-4893-a424-2c7fc3c1851d.png
│   │   │   └── ...
│   │   ├── ...
│   ├── ilid.json
```

3. Run

```
   python scr/main.py
```

4. Thanks to the author of the paper "Industrial Language-Image Dataset (ILID): Adapting Vision Foundation Models for Industrial Settings" for providing the ILID dataset.
