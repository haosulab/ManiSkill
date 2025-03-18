# mani_skill.sim package

This submodule contains all code for various other simulation systems, currently of which only MPM is included.

These systems are designed the same way SAPIEN is designed and put here (instead of in SAPIEN) for easy hacking, integration, and flexibility. SAPIEN's entity component system design is followed here so to easily leverage other features of SAPIEN with respect to these specific systems (like SAPIEN rendering, rigid body physics etc.)


- `mpm/` contains code for the MPM system.