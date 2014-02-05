# Amira Script

proc initScript {} {
    global inputVolumePath
    global outputVolumePath
    global patchSize
    global pcaDims
    global dataStdDev
    global spatialStdDev
    global simulationFlag

    # Clean workspace
    # remove -all

    # Read values in environment variables
    set inputVolumePath [app getenv AMIRA_DENOISE_GKDTREES_INPUT_VOLUME]
    set outputVolumePath [app getenv AMIRA_DENOISE_GKDTREES_OUTPUT_VOLUME]
    set patchSize [app getenv AMIRA_DENOISE_GKDTREES_PATCH_SIZE]
    set pcaDims [app getenv AMIRA_DENOISE_GKDTREES_PCA_VALUE]
    set dataStdDev [app getenv AMIRA_DENOISE_GKDTREES_DATA_STDDEV_VALUE]
    set spatialStdDev [app getenv AMIRA_DENOISE_GKDTREES_SPATIAL_STDDEV_VALUE]
    set simulationFlag [app getenv AMIRA_DENOISE_GKDTREES_SIMULATION]

    # Print values
    echo "** Denoising with parameters:"
    echo "** Input path: $inputVolumePath"
    echo "** Output path: $outputVolumePath"
    echo "** Patch size: $patchSize"
    echo "** PCA dims: $pcaDims"
    echo "** Data std. dev.: $dataStdDev"
    echo "** Spatial std. dev: $spatialStdDev"
    echo "** Simulation Flag: $simulationFlag"
}


proc loadVolume {} {
    global inputVolumePath
    global inputVolume

    set inputVolume [load $inputVolumePath]
}

proc castVolume {} {
    global inputVolume

    # Get value range of the input volume
    set inVolumeMinMax [$inputVolume getRange]
    lassign $inVolumeMinMax inVolumeMin inVolumeMax

    # Set value range of the cast volume
    set outVolumeMin 0.0
    set outVolumeMax 255.0

    # Properly set scale and offset from the parameters above
    set scale  [expr ($outVolumeMax - $outVolumeMin) / ($inVolumeMax - $inVolumeMin)]
    set offset [expr ($outVolumeMin / $scale) - $inVolumeMin]

    # Generate cast field and set port values
    set castField [create HxCastField]
    $castField data connect $inputVolume
    $castField outputType setValue 0
    $castField scaling setValue 0 $scale
    $castField scaling setValue 1 $offset

    # Generate cast volume data
    $castField action hit
    $castField fire

    # Set input volume as the cast one
    set inputVolume [$castField getResult]
}


proc startDenoising {} {
    global patchSize
    global windowSize
    global patchSize
    global pcaDims
    global dataStdDev
    global spatialStdDev
    global simulationFlag
    global inputVolume
    global outputVolume

    echo "** Generating input and output directories for intermediate data..."
    set tmpDir "/tmp/gkdtree-amira-denoising"
    set inputDir  "$tmpDir/$inputVolume/input"
    set outputDir "$tmpDir/$inputVolume/output"

    system "rm -rf $inputDir"
    system "rm -rf $outputDir"
    system "mkdir -p $inputDir"
    system "mkdir -p $outputDir"

    echo "** Saving input volume as a set of PNG files..."
    $inputVolume save PNG "$inputDir/$inputVolume"

    # Hack-sort-of: fix error in saving data. For some reason, when saving
    # in PNG format Amira does not correctly save the first two files. I
    # try to create them and also remove unused files.
    system "cp $inputDir/$inputVolume.png $inputDir/${inputVolume}0000.png"
    system "cp $inputDir/$inputVolume.png $inputDir/${inputVolume}0001.png"
    system "rm -f $inputDir/$inputVolume.png"
    system "rm -f $inputDir/$inputVolume.info"

    echo "** Changing color space in PNG images to gray..."
    system "mogrify -colorspace Gray $inputDir/*.png"

    echo "** Executing denoising script..."
    set scriptPath "/vis/data/people/bzflamas/AmmoniteDenoising/gkdtrees/scripts/process_volume.sh"

    if {!$simulationFlag} {
        set timeResolution seconds
        set startTime [clock $timeResolution]

        exec $scriptPath $inputDir $outputDir $pcaDims $patchSize $patchSize $patchSize $dataStdDev $spatialStdDev $spatialStdDev 1

        set endTime [clock $timeResolution]
        set elapsedTime [expr $endTime-$startTime]
        echo "** Time required to denoise (seconds): $elapsedTime"

        # Retrieve result from disk
        set outputFile $outputDir/output.info
        set outputVolume [load $outputFile]
    } else {
        echo "** Simulating denoising with script at $scriptPath"
        set outputVolume [$inputVolume duplicate]
    }
}


proc saveVolume {} {
    global outputVolumePath
    global outputVolume

    $outputVolume save {Amiramesh binary} $outputVolumePath
}


proc main {} {
    set hideNewModules 1

    initScript
    loadVolume
    castVolume
    startDenoising
    saveVolume
}

main
