ManiSkill2
=======================================

ManiSkill is a feature-rich GPU-accelerated robotics benchmark built on top of SAPIEN_ designed to provide accessible support for a wide array of applications from robot learning, learning from demonstrations, sim2real/real2sim, and more. 

Features:

* GPU parallelized simulation enabling 200,000+ FPS on some tasks
* GPU parallelized rendering enabling 10,000+ FPS on some tasks, massively outperforming other benchmarks
* Flexible API to build custom tasks of any complexity
* Variety of verified robotics tasks with diverse dynamics and visuals
* Reproducible baselines in Reinforcement Learning and Learning from Demonstrations, spread across tasks from dextrous manipulation to mobile manipulation 


.. toctree::
   :caption: Getting Started
   :maxdepth: 1

   getting_started/installation
   getting_started/quickstart
   getting_started/docker
   

.. toctree::
   :maxdepth: 1
   :caption: Resources

   tutorials/index
   concepts/index
   datasets/index
   algorithms_and_models/index
   workflows/index
   benchmark/online_leaderboard

.. toctree::
   :maxdepth: 2
   :caption: Additional Resources

   additional_resources/performance_benchmarking
   additional_resources/education

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
.. _SAPIEN: https://github.com/haosulab/sapien