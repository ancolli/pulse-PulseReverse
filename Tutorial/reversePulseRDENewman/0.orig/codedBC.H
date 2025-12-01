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
   
    //const scalar t = this->db().time().value();
    //const scalar yVel = 0.001;

    vectorField p(this->patch().localPoints());
    vectorField displacement(p.size(),vector(0,0,0));
    
    const polyMesh& mesh = this->internalField().mesh()();
    const label& patchID = this->patch().index();
    const vectorField faceNorm = -mesh.boundaryMesh()[patchID].faceNormals(); 
    //vectorField faceNorm(p.size(),vector(0,1,0));
    
    //const volScalarField& field = this->db().objectRegistry::lookupObject<volScalarField>("fi");
    //const volScalarField& keff = this->db().objectRegistry::lookupObject<volScalarField>("keff");
    const volScalarField& rho_i = this->db().objectRegistry::template lookupObject<volScalarField>("rho_i"); //this->db().objectRegistry::lookupObject<volScalarField>("rho_i");//
    
    const dictionary& kineticProperties = db().lookupObject<IOdictionary>("transportProperties");
    const scalar Dmol = readScalar(kineticProperties.lookup("Dmol"));
    
    scalarField Flux = -(63.5/8960/1000)*Dmol*rho_i.boundaryField()[patchID].snGrad(); //-(63.5/8960/1000/96485/2.0)*keff.boundaryField()[patchID]*field.boundaryField()[patchID].snGrad(); //m/s /1000
    
    //Info << "Flux = " <<  Flux << endl;

    PrimitivePatchInterpolation<primitivePatch> patchInterpolator (rho_i.mesh().boundaryMesh()[patchID]); //<primitivePatch>
    
    //- Perform interpolation 
    vectorField FluxPointValues = patchInterpolator.faceToPointInterpolate(Flux*faceNorm); //scalarField FluxPointValues = patchInterpolator.faceToPointInterpolate(fluxFaceValues);  
     
    //scalar dx2 = 1e-5; //scale = 0.001, MinBump = 0.05, MaxBump = 0.15, 
    
    //scalar MaxBump = max(p.component(0));
    //scalar MinBump = min(p.component(0));
        
    //scalar MaxParallel = returnReduce(MaxBump, maxOp<scalar>());
    //scalar MinParallel = returnReduce(MinBump, minOp<scalar>());
   
    displacement = FluxPointValues; //*t
    
    Info << "displacement = " <<  max(mag(displacement)) << endl;
    
    forAll(p, idx) 
    {
    	//displacement[idx].y() = (p[idx].x() == MinParallel || p[idx].x() == MaxParallel || p[idx].y() < 0.0 ) ? vector(0,0,0).y() : FluxPointValues[idx].y();
    	
    	//displacement[idx].x() = (p[idx].x() == MinParallel || p[idx].x() == MaxParallel) ? vector(0,0,0).x() : FluxPointValues[idx].x();
    	
    	//(p[idx].x() <= (MinParallel + dx2) || p[idx].x() >= (MaxParallel - dx2)) ? vector(0,0,0).y() : FluxPointValues[idx].y();  //*t 
    	//displacement[idx].y() = ( edge.start() == p[idx].x() || edge.end() == p[idx].x()) ? vector(0,0,0).y() : FluxPointValues[idx].y();  //*t 
    	//displacement[idx].x() = FluxPointValues[idx].x(); //*t   
    	//displacement[idx].y() = (displacement[idx].y() < 0.0 ) ? vector(0,0,0).y() : displacement[idx].y(); 	
    } 
    
   
    //Pout << "displacement = " <<  displacement << endl;
    
    operator==(displacement);
    
#};
   
codeOptions
#{
    -I$(WM_PROJECT_USER_DIR)/applications/lnInclude
#};
