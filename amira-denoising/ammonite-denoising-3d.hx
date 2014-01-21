proc getStrIterVal {iterVal} {
    return [format {%03d} $iterVal]
}

proc getStrWindowVal {windowVal} {
    return [format {%04d} $windowVal]
}

proc getStrSimilVal {similVal} {
    return [format {%02d} [expr int($similVal*10)]]
}

proc getVolName {iterVal windowVal similVal} {
    set strIterVal [getStrIterVal $iterVal]
    set strWindowVal [getStrWindowVal $windowVal]
    set strSimilVal [getStrSimilVal $similVal]

    return iter-$strIterVal-win-$strWindowVal-simil-$strSimilVal
}

proc executeNMLFilter {volume windowVal similarityVal iterVal} {
    # Open an editor for the current volume
    $volume setEditor [create HxImageVisionEditor]
    set currFilter [$volume getEditor]

    # Go to the NLM denoising filter
    $currFilter filter setValue 14
    $currFilter fire

    # Change the parameters of the filter
    $currFilter filter setValue 1 3
    $currFilter searchWindow setValue $windowVal
    $currFilter similarityValue setValue $similarityVal

    # Start filtering
    $currFilter doIt hit
    $currFilter fire
}

proc denoiseVolume {inVolume inOrthoSlice iterVal windowVal similVal} {
    # Generate a copy of the original volume and orthoslice and hide
    # the orthoslice
    set outVolume [$inVolume duplicate]
    set outOrthoSlice [$inOrthoSlice duplicate]

    $outOrthoSlice data connect $outVolume
    $outOrthoSlice frameSettings setValue 0 0
    $outOrthoSlice sliceOrientation setValue 2
    $outOrthoSlice options setValue 0 1
    $outOrthoSlice fire

    # Denoise volume
    executeNMLFilter $outVolume $windowVal $similVal $iterVal

    # Change name of the volume
    set currVolName [getVolName $iterVal $windowVal $similVal]
    $outVolume setLabel $currVolName

    # Show orthoslice
    $outOrthoSlice setViewerMask 1
    viewer 0 redraw
    $outOrthoSlice setViewerMask 0
}

# Remove all objects from the pool
remove -all

# Set a unique viewer
viewer setVertical 0

viewer 0 setBackgroundMode 1
viewer 0 setBackgroundColor 0 0.0980392 0.298039
viewer 0 setBackgroundColor2 0.686275 0.701961 0.807843
viewer 0 setTransparencyType 5
viewer 0 setAutoRedraw 0
viewer 0 show
mainWindow show

# Set a parallel perspective in the viewer
viewer 0 setCameraType orthographic

# Load original volume
set volPath {/media/data/bzflamas/AmmoniteDenoising/datasets/volumes/Ammonit-Nano-CT/amira/center-141x151x180/Ammonit-Eo_u-cropped-141x151x180-XY.vol.am}
set origVolume [load $volPath]

# Load OrthoSlice module and associate it to the original volume
set origOrthoSlice [create HxOrthoSlice]
$origOrthoSlice data connect $origVolume

# Set slice orientation as yz and automatically adjust view
$origOrthoSlice frameSettings setValue 0 0
$origOrthoSlice sliceOrientation setValue 2
$origOrthoSlice options setValue 0 1
$origOrthoSlice fire

# Show orthoslice
$origOrthoSlice setViewerMask 1
viewer 0 redraw
$origOrthoSlice setViewerMask 0

# Denoise the same volume with several combinations of parameter values. For
# each parameter, up to 3 denoisings are done on the same volume (using the
# previous results to avoid repeating computations).
for {set iterVal 1} {$iterVal <= 3} {set iterVal [expr $iterVal+1]} {
    foreach windowVal {11 21 31} {
        foreach similVal {0.5 0.6 0.7 0.8 0.9 1.0} {
            if {$iterVal == 1} {
                set srcVolume $origVolume
            } else {
                set srcVolume [getVolName [expr $iterVal-1] $windowVal $similVal]
            }

            denoiseVolume $srcVolume $origOrthoSlice $iterVal $windowVal $similVal
        }
    }
}

# Redraw viewer
viewer 0 setAutoRedraw 1
viewer 0 redraw
