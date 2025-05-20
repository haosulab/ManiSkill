# Inspire Hand

This is the folder for the dexterous Inspire Robotics Hands. It has a CC BY-NC-SA 4.0 license. The URDF is based on the one provided by the Inspire Robotics support team (which is not publicly available at the time of writing).

Currently the RH56DFX-2L/R with wrist model is supported.

Changes Made:
- Tuned the joint axes/signs to match the real robot.
- Tuned the limits of some joints to match the current multipliers and offsets
- Added a floating base version of the inspire hand.

## Notes

### Current Issues

There are still some small issues with the provided URDF from Inspire Robotics. They are noted below, to be removed once fixed:
- The mimic joint offsets are not tuned, needs system ID.
- Some joint limits are not tuned for mimic joints, need system ID. Currently there is some extra space in the limits that should be shrunk once offsets are identified.

### Simulation

It is unclear why but for the physx backends a damping of 0.001 (or some small value) needed to be set on the mimic joints for them to simulate stably. This is done at the controller level at the moment in ManiSkill.