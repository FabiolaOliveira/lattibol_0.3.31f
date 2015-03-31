precision highp float;
uniform sampler3D VOLUME;   //volume 3D com os valores
uniform sampler1D COLOR;    //volume 1D com as falsas cores
uniform float minU;
uniform float maxU;

void main(void) 
{
	vec3 values = texture(VOLUME, gl_TexCoord[0].xyz).rgb; // read velocity
	float value = length(values);
	
	if (value == 0.0) gl_FragColor = 0; 
	else {
		value = (value-minU)/(maxU-minU);
		if (value < 0.001953125) gl_FragColor =  texture1D(COLOR, 0.001953125);
		else if (value > 0.998046875) gl_FragColor =  texture1D(COLOR, 0.998046875);
		else gl_FragColor = texture(COLOR, value);
	}
}
