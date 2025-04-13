<!-- THIS IS ALL GENERATED DOCUMENTATION. DO NOT MODIFY THIS FILE -->
[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg
[dense-reward-badge]: https://img.shields.io/badge/dense%20reward-yes-green.svg
[sparse-reward-badge]: https://img.shields.io/badge/sparse%20reward-yes-green.svg
[no-dense-reward-badge]: https://img.shields.io/badge/dense%20reward-no-red.svg
[no-sparse-reward-badge]: https://img.shields.io/badge/sparse%20reward-no-red.svg
[demos-badge]: https://img.shields.io/badge/demos-yes-green.svg
# Drawing Tasks

These are tasks where the robot is controlled to draw a specific shape or pattern.
The document here has both a high-level overview/list of all tasks in a table as well as detailed task cards with video demonstrations after.

## Task Table
Table of all tasks/environments in this category. Task column is the environment ID, Preview is a thumbnail pair of the first and last frames of an example success demonstration. Max steps is the task's default max episode steps, generally tuned for RL workflows.
<table class="table">
<thead>
<tr class="row-odd">
<th class="head"><p>Task</p></th>
<th class="head"><p>Preview</p></th>
<th class="head"><p>Dense Reward</p></th>
<th class="head"><p>Success/Fail Conditions</p></th>
<th class="head"><p>Demos</p></th>
<th class="head"><p>Max Episode Steps</p></th>
</tr>
</thead>
<tbody>
<tr class="row-odd">
<td><p><a href="#tabletopfreedraw-v1">TableTopFreeDraw-v1</a></p></td>
<td><div style='display:flex;gap:4px;align-items:center'><img style='min-width:min(50%, 100px);max-width:100px;height:auto' src='../../_static/env_thumbnails/TableTopFreeDraw-v1_rt_thumb_first.png' alt='TableTopFreeDraw-v1'> <img style='min-width:min(50%, 100px);max-width:100px;height:auto' src='../../_static/env_thumbnails/TableTopFreeDraw-v1_rt_thumb_last.png' alt='TableTopFreeDraw-v1'></div></td>
<td><p>❌</p></td>
<td><p>❌</p></td>
<td><p>❌</p></td>
<td><p>1000</p></td>
</tr>
<tr class="row-odd">
<td><p><a href="#drawsvg-v1">DrawSVG-v1</a></p></td>
<td><div style='display:flex;gap:4px;align-items:center'><img style='min-width:min(50%, 100px);max-width:100px;height:auto' src='../../_static/env_thumbnails/DrawSVG-v1_rt_thumb_first.png' alt='DrawSVG-v1'> <img style='min-width:min(50%, 100px);max-width:100px;height:auto' src='../../_static/env_thumbnails/DrawSVG-v1_rt_thumb_last.png' alt='DrawSVG-v1'></div></td>
<td><p>❌</p></td>
<td><p>✅</p></td>
<td><p>❌</p></td>
<td><p>500</p></td>
</tr>
<tr class="row-odd">
<td><p><a href="#drawtriangle-v1">DrawTriangle-v1</a></p></td>
<td><div style='display:flex;gap:4px;align-items:center'><img style='min-width:min(50%, 100px);max-width:100px;height:auto' src='../../_static/env_thumbnails/DrawTriangle-v1_rt_thumb_first.png' alt='DrawTriangle-v1'> <img style='min-width:min(50%, 100px);max-width:100px;height:auto' src='../../_static/env_thumbnails/DrawTriangle-v1_rt_thumb_last.png' alt='DrawTriangle-v1'></div></td>
<td><p>❌</p></td>
<td><p>✅</p></td>
<td><p>✅</p></td>
<td><p>300</p></td>
</tr>
</tbody>
</table>

## TableTopFreeDraw-v1

![no-dense-reward][no-dense-reward-badge]
![no-sparse-reward][no-sparse-reward-badge]
:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Instantiates a table with a white canvas on it and a robot with a stick that draws red lines. This environment is primarily for a reference / for others to copy
to make their own drawing tasks.

**Randomizations:**
None

**Success Conditions:**
None

**Goal Specification:**
None
:::

<div style="display: flex; justify-content: center;">
<video preload="none" controls="True" width="100%" style="max-width: min(100%, 512px);" poster="../../_static/env_thumbnails/TableTopFreeDraw-v1_rt_thumb_first.png">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/TableTopFreeDraw-v1_rt.mp4" type="video/mp4">
</video>
</div>

## DrawSVG-v1

![no-dense-reward][no-dense-reward-badge]
![sparse-reward][sparse-reward-badge]
:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Instantiates a table with a white canvas on it and a svg path specified with an outline. A robot with a stick is to draw the triangle with a red line.

**Randomizations:**
- the goal svg's position on the xy-plane is randomized
- the goal svg's z-rotation is randomized in range [0, 2 $\pi$]

**Success Conditions:**
- the drawn points by the robot are within a euclidean distance of 0.05m with points on the goal svg
:::

<div style="display: flex; justify-content: center;">
<video preload="none" controls="True" width="100%" style="max-width: min(100%, 512px);" poster="../../_static/env_thumbnails/DrawSVG-v1_rt_thumb_first.png">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/DrawSVG-v1_rt.mp4" type="video/mp4">
</video>
</div>

## DrawTriangle-v1

![no-dense-reward][no-dense-reward-badge]
![sparse-reward][sparse-reward-badge]
![demos][demos-badge]
:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Instantiates a table with a white canvas on it and a goal triangle with an outline. A robot with a stick is to draw the triangle with a red line.

**Randomizations:**
- the goal triangle's position on the xy-plane is randomized
- the goal triangle's z-rotation is randomized in range [0, 2 $\pi$]

**Success Conditions:**
- the drawn points by the robot are within a euclidean distance of 0.05m with points on the goal triangle
:::

<div style="display: flex; justify-content: center;">
<video preload="none" controls="True" width="100%" style="max-width: min(100%, 512px);" poster="../../_static/env_thumbnails/DrawTriangle-v1_rt_thumb_first.png">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/DrawTriangle-v1_rt.mp4" type="video/mp4">
</video>
</div>
