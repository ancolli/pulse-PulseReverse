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
     
        const volScalarField& kf_ = this->db().objectRegistry::template lookupObject<volScalarField>("keff");   // Desired field 1
        const volScalarField& fi_ = this->db().objectRegistry::template lookupObject<volScalarField>("fi"); // Desired field 2

        scalarField kf_boundary = kf_.boundaryField()[patchID];                                              
        scalarField fi_boundary = fi_.boundaryField()[patchID];       
        scalarField grad_fi = fi_.boundaryField()[patchID].snGrad();                                     

        scalarField gradiente = kf_boundary/0.5*grad_fi*(0-fi_boundary); 
        //Info<< "gradiente:" << gradiente << endl;
        
        this->refValue() = 0;//B/A;
        this->refGrad() = gradiente;
        this->valueFraction() = 0;//A/(A+1); 
#};
   
codeOptions
#{
    -I$(WM_PROJECT_USER_DIR)/applications/lnInclude
#};
