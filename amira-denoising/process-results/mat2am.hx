# Amira Script

proc initScript {} {
    global inputVolumePath
    global outputVolumePath

    set inputVolumePath [app getenv AMIRA_MATTOAM_INPUT_VOLUME]
    set outputVolumePath [app getenv AMIRA_MATTOAM_OUTPUT_VOLUME]
}


proc loadVolume {} {
    global inputVolumePath
    global inputVolume

    set inputVolume [load $inputVolumePath]
}


proc saveVolume {} {
    global outputVolumePath
    global inputVolume

    $inputVolume save {Amiramesh binary} $outputVolumePath
}


proc main {} {
    set hideNewModules 1

    initScript
    loadVolume
    saveVolume
}

main
