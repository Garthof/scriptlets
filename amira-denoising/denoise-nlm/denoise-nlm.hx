# Amira Script

proc initScript {} {
    global inputVolumePath
    global outputVolumePath
    global patchSize
    global windowSize
    global similarity
    global mode
    global simulationFlag
    global volume

    # Clean workspace
    # remove -all

    # Read values in environment variables
    set inputVolumePath [app getenv AMIRA_DENOISE_NLM_INPUT_VOLUME]
    set outputVolumePath [app getenv AMIRA_DENOISE_NLM_OUTPUT_VOLUME]
    set patchSize [app getenv AMIRA_DENOISE_NLM_PATCH_SIZE]
    set windowSize [app getenv AMIRA_DENOISE_NLM_WINDOW_VALUE]
    set similarity [app getenv AMIRA_DENOISE_NLM_SIMILARITY_VALUE]
    set mode [app getenv AMIRA_DENOISE_NLM_MODE]
    set simulationFlag [app getenv AMIRA_DENOISE_NLM_SIMULATION]

    # Print values
    echo "** Denoising with parameters:"
    echo "** Input path: $inputVolumePath"
    echo "** Output path: $outputVolumePath"
    echo "** Patch size: $patchSize"
    echo "** Window size: $windowSize"
    echo "** Similarity: $similarity"
    echo "** Mode: $mode"
    echo "** Simulation Flag: $simulationFlag"
}


proc loadVolume {} {
    global inputVolumePath
    global volume

    set volume [load $inputVolumePath]
}


proc startDenoising {} {
    global patchSize
    global windowSize
    global similarity
    global mode
    global simulationFlag
    global volume

    # Open an editor for the current volume
    $volume setEditor [create HxImageVisionEditor]
    set editor [$volume getEditor]

    # Go to the NLM denoising filter
    $editor filter setValue 14
    $editor fire

    # # Change the parameters of the filter
    set formattedMode [string toupper $mode]

    if {$formattedMode eq "2D"} {
        $editor filter setValue 1 2
    } elseif {$formattedMode eq "3D"} {
        $editor filter setValue 1 3
    } else {
        error "** Error: $mode is not a valid mode"
    }

    $editor neighborhood setValue $patchSize
    $editor searchWindow setValue $windowSize
    $editor similarityValue setValue $similarity
    $editor cudaDevice setValue 1
    $editor doIt hit

    # Start filtering
    if {!$simulationFlag} {
        set timeResolution seconds
        set startTime [clock $timeResolution]

        $editor fire

        set endTime [clock $timeResolution]
        set elapsedTime [expr $endTime-$startTime]
        echo "** Time required to denoise (seconds): $elapsedTime"
    } else {
        echo "** Simulating denoising"
    }
}


proc saveVolume {} {
    global outputVolumePath
    global volume

    $volume save {Amiramesh binary} $outputVolumePath
}


proc main {} {
    set hideNewModules 1

    initScript
    loadVolume
    startDenoising
    saveVolume
}

main
