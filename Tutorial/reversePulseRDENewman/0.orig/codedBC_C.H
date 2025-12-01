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
       //vectorField p(this->patch().localPoints());
        //const polyMesh& mesh = this->internalField().mesh()();
        const label& patchID = this->patch().index();
        
        
        const fvPatch& p = this->patch();                                                  // Actual boundary/patch
    	//const label patchID = p.patch().boundaryMesh().findPatchID("wall_WE1");            // Desired patch ID
        const fvMesh& mesh = patch().boundaryMesh().mesh();
	  
        const volScalarField& fi_ = this->db().objectRegistry::template lookupObject<volScalarField>("fi");                                 
        scalarField fi_boundary = fi_.boundaryField()[patchID]; 
        
        const volScalarField& Con_ox = this->db().objectRegistry::template lookupObject<volScalarField>("rho_i");   // Desired field 3       
        scalarField C_boundary = Con_ox.boundaryField()[patchID];                    
	
        scalarField delta = 1.0/mesh.deltaCoeffs().boundaryField()[patchID]; 

        const dictionary& kineticProperties = db().lookupObject<IOdictionary>("transportProperties");

	scalar nu_e = readScalar(kineticProperties.lookup("nu_e"));
        
        scalar j0 = readScalar(kineticProperties.lookup("j0_"));
        scalar bc = readScalar(kineticProperties.lookup("bc_"));
	scalar ba= readScalar(kineticProperties.lookup("ba_"));
        scalar C_inf = readScalar(kineticProperties.lookup("rho_i_inf"));
        
        
        //dimensionedScalar D_Cu(kineticProperties.lookup("Dmol"));
        scalar D_Cu = readScalar(kineticProperties.lookup("Dmol"));	
        scalar F = readScalar(kineticProperties.lookup("F"));
        scalar t = readScalar(kineticProperties.lookup("t"));
 
        scalarField jota_c = -j0*pow((C_boundary/C_inf),0.42)*exp(-fi_boundary/bc);                                     
        scalarField jota_a = j0*pow((C_boundary/C_inf),0.42)*exp(-fi_boundary/ba);
	scalarField k0 = D_Cu/delta;       //.value()                                                        
        
	scalarField a = -0.42/nu_e*(jota_c+jota_a)/C_boundary/F/k0;        
	scalarField b = (1-0.42)/nu_e*(jota_c+jota_a)/F/k0;                                     
	                  
        this->refValue() = b/a;
        this->refGrad() = 0;
        this->valueFraction() = a/(1+a);
#};
   
codeOptions
#{
    -I$(WM_PROJECT_USER_DIR)/applications/lnInclude
#};
