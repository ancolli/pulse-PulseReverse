//{
//const word control = word(controlProperties.lookup("pc"));

//const scalar Ei = readScalar(controlProperties.lookup("Ei"));
//const scalar Ef = readScalar(controlProperties.lookup("Ef"));

//const scalar v = readScalar(controlProperties.lookup("v"));
//const scalar Tf = readScalar(controlProperties.lookup("Tf"));
//const scalar Tf = readScalar(controlDict.lookup("endTime"));
//const scalar n = readScalar(controlProperties.lookup("n"));
const scalar maxPot = readScalar(controlProperties.lookup("maxPot"));
const scalar Irev = readScalar(controlProperties.lookup("Irev"));
const scalar Ifwd = readScalar(controlProperties.lookup("Ifwd"));

const scalar trev = readScalar(controlProperties.lookup("trev"));
const scalar tfwd = readScalar(controlProperties.lookup("tfwd"));
const scalar ciclo = tfwd + trev;
const scalar tiempo = mesh.time().value()-1e-12;

//const scalar Tf_ciclo = Tf/n;

//const scalar i = floor(tiempo/Tf_ciclo);

//const scalar t = tiempo-i*Tf_ciclo;
scalar I = 0;
scalar pot = 0;
bool T_rev = 0;
  
const scalar tiempoEnCiclo = fmod(tiempo, ciclo); // calcula la posición del tiempo dentro del ciclo actual

if (tiempoEnCiclo < tfwd) {
    I = Ifwd;
    pot = maxPot;
    T_rev = 0;
    
} else {
    I = Irev;
    pot = -maxPot;
    T_rev = 1;
}



// if ( control == "linear") {
        //potential field
	//potencial = (t < Tf_ciclo/2) ? Ei + t*v : Ef - (t-Tf_ciclo/2)*v ;//
	
//} else if (control == "ramp") {

	//potencial = (t <= Tf_ciclo/2) ? Ei : Ef ;//
//}

//controlProperties.set("EE", potencial);
//Info << "Potencial = " << potencial << " V" << endl;
//}

// ************************************************************************* //
