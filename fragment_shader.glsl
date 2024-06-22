
#version 330 core
out vec4 FragColor;

uniform float radius;
uniform vec2 center;
uniform vec2 resolution;

void main()
{
    vec2 uv = (gl_FragCoord.xy - center) / resolution;
    float dist = length(uv) * resolution.y;  // Adjusted for pixel space
    if (dist <= radius) {
        FragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red circle
    } else {
        discard;  // Ignore pixels outside the radius
    }
}
