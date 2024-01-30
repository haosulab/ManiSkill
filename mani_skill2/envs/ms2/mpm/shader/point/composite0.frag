#version 450

layout(set = 0, binding = 0) uniform sampler2D samplerPointDepthLinear;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outSmoothedDepthLinear;

layout(set = 1, binding = 0) uniform CameraBuffer {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
  mat4 prevViewMatrix;
  mat4 prevViewMatrixInverse;
  float width;
  float height;
} cameraBuffer;

float sqr(float x) {
  return x * x;
}

// void main() {
//   outSmoothedDepthLinear.x = texture(samplerPointDepthLinear, inUV).x;
// }

void main()
{
  vec2 res = vec2(cameraBuffer.width, cameraBuffer.height);
  vec2 coord = vec2(gl_FragCoord.xy);
  float depth = texture(samplerPointDepthLinear, coord / res).x;

	if (depth == 0.0)
	{
    outSmoothedDepthLinear.x = 0.0;
		return;
	}

	float blurDepthFalloff = 5.5;
  // float radius = 10.0;
	float radius = min(3.0 / (-depth), 16);
	float radiusInv = 1.0/radius;
	float taps = ceil(radius);
	float frac = taps - radius;

	float sum = 0.0;
  float wsum = 0.0;
	float count = 0.0;

  for(float y=-taps; y <= taps; y += 1) {
    for(float x=-taps; x <= taps; x += 1) {
      vec2 offset = vec2(x, y);
      float samp = texture(samplerPointDepthLinear, (coord + offset) / res).x;

      if (samp == 0.0) {
        continue;
      }

      // spatial domain
      float r1 = length(vec2(x, y))*radiusInv;
      float w = exp(-(r1*r1));

      // range domain (based on depth difference)
      float r2 = (samp - depth) * blurDepthFalloff;
      float g = exp(-(r2*r2));

			//fractional radius contributions
			float wBoundary = step(radius, max(abs(x), abs(y)));
			float wFrac = 1.0 - wBoundary*frac;

			sum += samp * w * g * wFrac;
			wsum += w * g * wFrac;
			count += g * wFrac;
    }
  }

  if (wsum > 0.0) {
    sum /= wsum;
  }

	float blend = count/sqr(2.0*radius+1.0);
  outSmoothedDepthLinear.x = mix(depth, sum, blend);
}
