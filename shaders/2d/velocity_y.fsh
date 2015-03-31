precision highp float;
uniform sampler2D VOLUME;   //volume 3D com os valores
uniform sampler1D COLOR;    //volume 1D com as falsas cores
uniform float minU;
uniform float maxU;
uniform float minRho;
uniform float maxRho;

void main(void) 
{
	float value, scaledValue;
	vec2 values, pos;
	
	if(gl_TexCoord[0].y > 0.5) {
		pos = vec2(gl_TexCoord[0].x, gl_TexCoord[0].y-0.5);
		value = texture(VOLUME, pos).a; 
		scaledValue = (value-minRho)/(maxRho-minRho);
	}
	else {
		value = texture(VOLUME, gl_TexCoord[0].xy).g; // read velocity	
		scaledValue = (value-minU)/(maxU-minU);
	}

	if (value == 0.0) gl_FragColor = 0; 
	else {
		if (scaledValue < 0.001953125) gl_FragColor =  texture1D(COLOR, 0.001953125);
		else if (scaledValue > 0.998046875) gl_FragColor =  texture1D(COLOR, 0.998046875);
		else gl_FragColor = texture(COLOR, scaledValue);
	}
}
