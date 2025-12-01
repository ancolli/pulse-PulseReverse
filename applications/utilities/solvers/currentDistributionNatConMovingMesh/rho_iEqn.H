{
    alphat = turbulence->nut()/Sct;
    alphat.correctBoundaryConditions();

    volScalarField alphaEff_c("alphaEff_c", Dmol + alphat);

    fvScalarMatrix rho_iEqn
    (
        fvm::ddt(rho_i)
      + fvm::div(phi, rho_i)
      - fvm::laplacian(alphaEff_c, rho_i)
     ==
        fvOptions(rho_i)
    );

    rho_iEqn.relax();

    fvOptions.constrain(rho_iEqn);

    rho_iEqn.solve();

    fvOptions.correct(rho_i);
   
    Info<< "Min/max C" << ' ' << min(rho_i).value() << ' ' << max(rho_i).value() << endl;
    
    const scalar a = readScalar(geometryProperties.lookup("a"));
    const scalar b = readScalar(geometryProperties.lookup("b"));
    
    // Calculates flux on cathode
        Info<< "Calculating Charge" << endl;

	label patchi1 = mesh.boundaryMesh().findPatchID("wall_WE");
	//label patchi2 = mesh.boundaryMesh().findPatchID("wall_WE2");

        //Flux.boundaryFieldRef()[patchi] = -2*96485*alphaEff_c.boundaryFieldRef()[patchi]*rho_i.boundaryFieldRef()[patchi].snGrad();
	//Flux.boundaryFieldRef()[patchi] = -2*96485*alphaEff_c.boundaryFieldRef()[patchi]*rho_i.boundaryFieldRef()[patchi].snGrad();		    
// Calculates average Flux
        //Flux.boundaryField()[patchi] += ;

        Carga.boundaryFieldRef()[patchi1] += 2*96485*alphaEff_c.boundaryFieldRef()[patchi1]*rho_i.boundaryFieldRef()[patchi1].snGrad()*mesh.magSf().boundaryField()[patchi1]*runTime.deltaTValue();
        //Carga.boundaryFieldRef()[patchi2] += 2*96485*alphaEff_c.boundaryFieldRef()[patchi2]*rho_i.boundaryFieldRef()[patchi2].snGrad()*mesh.magSf().boundaryField()[patchi2]*runTime.deltaTValue();
        
	//scalar area1 = gSum(mesh.magSf().boundaryField()[patchi1]);
        scalar sumField1 = 0;
	//scalar sumField2 = 0;	
        //if (area > 0)
        {
        	sumField1 = gSum
                (
                	Carga.boundaryField()[patchi1]
                );// / area;
                /*sumField2 = gSum
                (
                	Carga.boundaryField()[patchi2]
                );// / area;*/
        }

        //Info<< "throwing power = " << (a/b)/(a/b + sumField2/sumField1 - 1)*100 << endl; 
        
        //Results.set("throwing_power", (a/b)/(a/b + sumField2/sumField1 - 1)*100); 
        
        scalar currentWE = 0;
        //scalar currentWE2 = 0;
        {
        	currentWE = gSum
                (
                	2*96485*alphaEff_c.boundaryFieldRef()[patchi1]*rho_i.boundaryFieldRef()[patchi1].snGrad()*mesh.magSf().boundaryField()[patchi1]
                );// / area;
                /*currentWE2  = gSum
                (
                	2*96485*alphaEff_c.boundaryFieldRef()[patchi2]*rho_i.boundaryFieldRef()[patchi2].snGrad()*mesh.magSf().boundaryField()[patchi2]
                );// / area;*/
        }
        
        Info<< "Effective current WE = " << currentWE << endl; 
        //Info<< "Effective current WE2 = " << currentWE2 << endl; 
        
}
