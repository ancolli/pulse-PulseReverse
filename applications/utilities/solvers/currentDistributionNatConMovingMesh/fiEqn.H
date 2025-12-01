{

bool repeat(false);
// check if wall_anode exist
string patchNameNeeded = "wall_CE";
scalar patchOK = 0;

    forAll(mesh.boundaryMesh(), p_ith)
    {     
      string patchName = mesh.boundaryMesh()[p_ith].name();	
      if (patchNameNeeded == patchName)  // checks if "wall_anode" is in the current patchName
      {
        patchOK = 1;
      }
    }  

if (patchOK==1)  // checks if "wall_anode" is in the current patchName
      {
// if wall_anode exists then: Calculates flux on the anode
	
	// reading and writing probeLocation (reference electrode)

        label cellNo = mesh.findCell(vector(controlProperties.lookup("refLocation")));
	
        //Results.set("Vref", Vref);

// calculate cell potential
        scalar mult = 0.0;
        scalar Vcell_old = 0.0;
        scalar var = 0.0;
         
        int control = readScalar(controlProperties.lookup("control"));
        //control = (Irev == 0 && T_rev == 1) ? 2 : control;
        //scalar I = readScalar(controlProperties.lookup("I"));
        //scalar maxPot = readScalar(controlProperties.lookup("maxPot"));
        scalar magMinPot = readScalar(controlProperties.lookup("magMinPot"));
        scalar Irev = readScalar(controlProperties.lookup("Irev"));
        scalar alpha = readScalar(controlProperties.lookup("alpha"));
        scalar tol_I = readScalar(controlProperties.lookup("tol_I"));

        scalar ba = readScalar(transportProperties.lookup("ba"));

// potentiostatic mode
	scalar ref = readScalar(controlProperties.lookup("ref"));
        scalar maxvar = readScalar(controlProperties.lookup("maxvar"));
        scalar tol_ref = readScalar(controlProperties.lookup("tol_ref"));       

	scalar Vcell = readScalar(Results.lookup("Vcell"));
	
	//Info<< "hasta aca 1" << endl;

	do {  // permite cumplir con modo galvanostatico o potenciostatico luego de cada timeStep
		
// Calculates flux on the anode and cathode
        label patchi_CE = mesh.boundaryMesh().findPatchID("wall_CE"); 
	label patchi_WE = mesh.boundaryMesh().findPatchID("wall_WE");
	//label patchi_WE2 = mesh.boundaryMesh().findPatchID("wall_WE2");
	//label patchi_S = mesh.boundaryMesh().findPatchID("wall_sphere");  
        Flux.boundaryFieldRef()[patchi_CE] = keff.boundaryFieldRef()[patchi_CE]*fi.boundaryFieldRef()[patchi_CE].snGrad(); //gradients are defined inwards
	Flux.boundaryFieldRef()[patchi_WE] = keff.boundaryFieldRef()[patchi_WE]*fi.boundaryFieldRef()[patchi_WE].snGrad(); //gradients are defined inwards
	//Flux.boundaryFieldRef()[patchi_WE2] = keff.boundaryFieldRef()[patchi_WE2]*fi.boundaryFieldRef()[patchi_WE2].snGrad();//Flux.boundaryFieldRef()[patchi_S] = keff.boundaryFieldRef()[patchi_S]*fi.boundaryFieldRef()[patchi_S].snGrad();
// Calculates average Flux
	scalar sumField_CE = 0.0;
	scalar sumField_WE = 0.0;
	//scalar sumField_WE2 = 0.0;
        {
        	sumField_CE = gSum
                (
                	mesh.magSf().boundaryField()[patchi_CE]
                      * Flux.boundaryField()[patchi_CE]
                );
		sumField_WE = gSum
                (
                	mesh.magSf().boundaryField()[patchi_WE]
                      * Flux.boundaryField()[patchi_WE]
                ); //+ gSum (
                   //mesh.magSf().boundaryField()[patchi_S]
                   //   * Flux.boundaryField()[patchi_S]
                   //   )
                /*sumField_WE2 = gSum
                (
                	mesh.magSf().boundaryField()[patchi_WE2]
                      * Flux.boundaryField()[patchi_WE2]
                );*/
        }
                
        Info<< "Total current at CE = " << sumField_CE << " A "<< endl;
	Info<< "Total current at WE = " << sumField_WE << " A "<< endl;
	//Info<< "Total current at WE2 = " << sumField_WE2 << " A "<< endl;	
	
// potentiostatic mode
	scalar Vref = fi[cellNo];
	var = (ref < 0.0) ? -ref-Vref : ref-(Vcell-Vref); // <0 cathodic overpotential, >0 anodic overpotential
        var = (var > 0.0) ? min(var,maxvar) : max(var,-maxvar);
        var = (mag(var/ref) > tol_ref) ? var : 0.0; // limito ref a x%
        var = (control==1) ? var : ((control==2) ? 0.0 : 0.0); //
        //Info<< "hasta aca 1" << endl;      
// galvanostatic
        if (Irev == 0 && T_rev == 1) {
        	        mult = (mag(mag(sumField_CE)) > 1e-10) ? (sumField_CE)/ba/mag(I) : 0.0; 
        	} else {
        		mult = (mag(mag(sumField_CE)-mag(I))/(mag(I)) > tol_I) ? -ba*(mag(sumField_CE)-mag(I))/I : 0.0; // limito corriente a x%	(sumField_A-I)/(I/ba)    (mag(sumField_CE)-mag(I))*Vcell/max(mag(I),mag(sumField_CE))/3.0
        //mult = (mag(sumField-I)/I > tol_I) ? (sumField-I)*Vcell/sumField/3.0 : 0.0; // limito corriente a x%   (sumField-I)/(I/ba)
        }
        
        //Info<< "hasta aca 2" << endl;
// overrelaxation       
        Vcell_old = Vcell;
        //mult=(maxPot < 0.0) ? -mult : mult; // 
	//Vcell = (control==1) ? Vcell+var : ((control==2) ? maxPot : Vcell-mult);
	Vcell = (control==1) ? Vcell+var : ((control==2) ? pot : Vcell-mult);
        Vcell = Vcell_old+alpha*(Vcell-Vcell_old);
        //Info<< "var:" << var << endl;
	//Vcell = (maxPot < 0.0) ? min(max(Vcell, maxPot),-magMinPot) : max(min(Vcell, maxPot),magMinPot); // limited potentials
	if (Irev == 0 && T_rev == 1) {
		//Vcell = (Vcell < 0.0) ? min(max(Vcell, pot),-magMinPot) : max(min(Vcell, pot),magMinPot);
	} else {
		Vcell = (pot < 0.0) ? min(max(Vcell, pot),-magMinPot) : max(min(Vcell, pot),magMinPot);
	}
	
	//Info<< "hasta aca 3" << endl;

// writing Vcell, I and Vref  

        scalar sumField_WE_old = readScalar(Results.lookup("I"));
             
        Results.set("Vcell", Vcell);
        Results.set("I", sumField_WE);//Results.set("I", sumField_WE1+sumField_WE2);//Results.set("I", sumField_CE);
	Results.set("Vref", (ref < 0.0) ? -Vref : (Vcell-Vref));//
	Results.set("mult", mult);
        Results.set("var", var); 
        
        //Info<< "hasta aca 4" << endl;
          
// potential field
//	if (Irev == 0 && T_rev == 1) {
//		fi = 0.0*fi;
//		fi.correctBoundaryConditions();
	
//	} else {

		fi.storePrevIter();
	    	solve
	    	(
	       		- fvm::laplacian(keff, fi)
	    	);
	    	fi.relax();
//    	}
    	
    	//fi.correctBoundaryConditions();
    	
    	Info<< "Vcell:" << Vcell << endl;
    	
    	Info<< "Vref:" << Vref << endl;

	Info<< "Min/max fi:" << min(fi).value() << ' ' << max(fi).value() << endl;
    	   	
    	switch(control) {
        	case 1: repeat = (var != 0.0) ? true : false;   break; // potentiostatic mode
        	//case 2: repeat = ((mag(sumField_CE+sumField_WE1+sumField_WE2)/mag(sumField_CE)) > tol_I) ? true : false; break; //fixed cell potential
        	case 2:
        	//if (Irev == 0 && T_rev == 1) {
		//	 repeat = ((mag(sumField_WE)-mag(sumField_WE_old)) > 1e-10) ? true : false; break; //fixed cell potential	
		//} else {
			 repeat = ((mag(sumField_CE+sumField_WE)/max(mag(sumField_CE),1e-10)) > tol_I) ? true : false; break; //fixed cell potential		
    		//}        	
        	//case 2: repeat = ((mag(sumField_CE+sumField_WE)/max(mag(sumField_CE),1e-10)) > tol_I) ? true : false; break; //fixed cell potential
        	case 3: repeat = (mult != 0.0) ? true : false;  break; // galvanostatic mode
        	default: Info<< "control = 1-potentiostatic mode" 
        			<< "control = 2-fixed cell potential"
        			<< "control = 3-galvanostatic mode"
        			<< endl
            			<< exit(FatalError); 
    	}
   		
    	} while (repeat);	
   }
   else
   {
	FatalErrorIn
        (
            " Foam::patchProbes::findElements(const fvMesh&)"
        )   << " No patch name: wall_CE found "
            << " the patch name: wall_CE should exist "
            << endl
            << exit(FatalError);     
   }
}

// ************************************************************************* //
