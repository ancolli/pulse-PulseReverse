//#include "$FOAM_CASE/system/blockMeshDict"

codeInclude
#{
    #include "addToRunTimeSelectionTable.H"
    #include "volFields.H"
    #include "pointPatchFields.H"
    #include "PrimitivePatchInterpolation.H"
    //#include "vectorList.H"
#};

code
#{
           //const polyMesh& mesh = this->internalField().mesh()();
        //vectorField p(this->patch().localPoints());
        //const label& patchID = this->patch().index();
        
        //vectorField p(this->patch().localPoints());
        //const polyMesh& mesh = this->internalField().mesh()();
        const label& patchID = this->patch().index();
        
        
        const fvPatch& p = this->patch();                                                  // Actual boundary/patch
    	//const label patchID = p.patch().boundaryMesh().findPatchID("wall_WE1");            // Desired patch ID
        const fvMesh& mesh = patch().boundaryMesh().mesh();

        const volScalarField& kf_ = this->db().objectRegistry::template lookupObject<volScalarField>("keff");   // Desired field 1
        const volScalarField& fi_ = this->db().objectRegistry::template lookupObject<volScalarField>("fi"); // Desired field 2
        scalarField fi_boundary = fi_.boundaryField()[patchID];
        scalarField fi_boundaryOld = fi_.oldTime().boundaryField()[patchID];//prevIter(). //fi_.oldTime()
        
        scalarField kf_boundary = kf_.boundaryField()[patchID];
    //    const dictionary& transProp = db().lookupObject<IOdictionary>("transportProperties");      
    //const dimensionedScalar kf_boundary(transProp.lookup("kf"));
       
        const volScalarField& Con_ox = this->db().objectRegistry::template lookupObject<volScalarField>("rho_i");   // Desired field 3       
        scalarField C_boundary = Con_ox.boundaryField()[patchID];                                                                 
                                        
        scalarField delta = 1.0/mesh.deltaCoeffs().boundaryField()[patchID]; 

        //const dictionary& kineticProperties = db().lookupObject<IOdictionary>("transportProperties");
  
        const dictionary& kineticProperties = db().lookupObject<IOdictionary>("transportProperties");
  
        scalar j0 = readScalar(kineticProperties.lookup("j0_"));
        scalar bc = readScalar(kineticProperties.lookup("bc_"));
	scalar ba = readScalar(kineticProperties.lookup("ba_"));

        scalar j0_H2= readScalar(kineticProperties.lookup("j0_H2"));
        scalar bc_H2 = readScalar(kineticProperties.lookup("bc_H2"));
	scalar E0_H2 = readScalar(kineticProperties.lookup("E0_H2"));
	
	scalar C_inf = readScalar(kineticProperties.lookup("rho_i_inf"));
	
	scalar cap = readScalar(kineticProperties.lookup("cap"));
	
	scalar dt = this ->db().time().deltaTValue();
			
        scalarField jota_c = -j0*pow((C_boundary/C_inf),0.42)*exp((-fi_boundary)/bc);                                                        
        scalarField jota_a = j0*pow((C_boundary/C_inf),0.42)*exp((-fi_boundary)/ba);
        scalarField jota_secondary = 0.0*j0_H2*exp((-fi_boundary-E0_H2)/bc_H2);
        
        scalarField k0 = kf_boundary/delta;    //.value()   
        
        Info<< "min/max fi_boundary:" << min(fi_boundary) << " " << max(fi_boundary) << endl;
        Info<< "min/max fi_boundaryOld:" << min(fi_boundaryOld) << " " << max(fi_boundaryOld) << endl;
                                                                       
        
	scalarField A = (1/k0)*(jota_c/bc+jota_a/ba+jota_secondary/bc_H2+cap/dt);       
	scalarField B = (1/k0)*((1+fi_boundary/bc)*jota_c+(1+fi_boundary/ba)*jota_a+(1+fi_boundary/bc_H2)*jota_secondary-cap/dt*(fi_boundary-2*fi_boundaryOld)); 
        
	this->refValue() = B/A;
        this->refGrad() = 0;
        this->valueFraction() = A/(A+1); 
    
#};
   
codeOptions
#{
    -I$(WM_PROJECT_USER_DIR)/applications/lnInclude
#};
