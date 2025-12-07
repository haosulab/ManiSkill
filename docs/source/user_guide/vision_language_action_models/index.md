# Vision Language Action Models

ManiSkill supports evaluating and pretraining vision language action models. Currently the following VLAs have been tested via the ManiSkill framework:

- [Octo](https://github.com/octo-models/octo)
- [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer)
- [RT-x](https://robotics-transformer-x.github.io/)

RDT-1B uses some of the ManiSkill demonstrations for pretraining data and evaluates by fine-tuning on some demonstrations on various ManiSkill tasks, see their [README](https://github.com/thu-ml/RoboticsDiffusionTransformer?tab=readme-ov-file#simulation-benchmark) for more details.

Octo and RT series of models are evaluated through various real2sim environments as part of the SIMPLER project, see their [README](https://github.com/simpler-env/SimplerEnv/tree/maniskill3) for details on how to run the evaluation setup.