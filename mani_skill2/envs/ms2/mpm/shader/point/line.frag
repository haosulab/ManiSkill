#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inPrevPosition;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec4 outLine;

void main() {
  outLine = inColor;
}
