proc castVolume {inputVolume} {
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

    # Set result, clean everything and return
    set outputVolume [$castField getResult]
    remove $castField

    return $outputVolume
}

proc takeSnapshot {volName strOrientation} {

    # Load OrthoSlice module and associate it to the original volume
    set orthoSlice [create HxOrthoSlice]
    $orthoSlice data connect $volName

    if {$strOrientation eq "xy"} {
        set orientation 0
    } elseif {$strOrientation eq "xz"} {
        set orientation 1
    } elseif {$strOrientation eq "yz"} {
        set orientation 2
    }

    # Remove frame, automatically adjust view and change orientation
    $orthoSlice setViewerMask 1
    $orthoSlice frameSettings setValue 0 0
    $orthoSlice options setValue 0 1
    $orthoSlice sliceOrientation setValue $orientation
    $orthoSlice fire

    # Create an image from the orthoslice view and save as PNG format
    $orthoSlice createImage image
    set castImage [castVolume image]
    $castImage save PNG /tmp/[file rootname $volName]-$strOrientation.png

    # Clean everything
    remove image
    remove $castImage
    remove $orthoSlice

    # viewer 0 redraw
    # viewer 0 snapshot -alpha /tmp/$volName-$strOrientation.png
}

proc main {} {
    # Take three snapshots of each uniform scalar field in the pool, each snapshot
    # with a different orientation
    foreach volume [all HxUniformScalarField3] {

        takeSnapshot $volume "xy"
        takeSnapshot $volume "xz"
        takeSnapshot $volume "yz"
    }
}

main
