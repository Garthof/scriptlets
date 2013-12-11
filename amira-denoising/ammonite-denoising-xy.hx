proc executeNMLFilter {volume windowVal similarityVal iterVal} {
    # Open an editor for the current volume
    $volume setEditor [create HxImageVisionEditor]
    set currFilter [$volume getEditor]

    # Go to the NLM denoising filter
    $currFilter filter setValue 14
    $currFilter fire

    # Change the parameters of the filter
    $currFilter searchWindow setValue $windowVal
    $currFilter similarityValue setValue $similarityVal

    # Start filtering
    $currFilter doIt hit
    $currFilter fire
}

proc denoiseVolume {inVolume inOrthoSlice windowVal similarityVal iterVal} {
    # Generate a copy of the original volume and orthoslice and hide
    # the orthoslice
    set outVolume [$inVolume duplicate]
    set outOrthoSlice [$inOrthoSlice duplicate]

    $outOrthoSlice data connect $inVolume
    $outOrthoSlice frameSettings setValue 0 0
    $outOrthoSlice sliceOrientation setValue 2
    $outOrthoSlice options setValue 0 1
    $outOrthoSlice fire

    # Denoise volume
    executeNMLFilter $outVolume $windowVal $similarityVal $iterVal

    # Change name of the volume
    set currVolName window-$windowVal-similarity-$similarityVal-iteration-$iterVal
    $outVolume setLabel $currVolName

    # Take a snapshot of the result and hide the volume and the orthoslice
    $outOrthoSlice setViewerMask 1
    viewer 0 redraw
    viewer 0 snapshot -alpha /tmp/$currVolName.png
    $outOrthoSlice setViewerMask 0
}

# Remove all object from the pool
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
set volPath {/media/data/bzflamas/AmmoniteDenoising/datasets/volumes/Ammonit-Nano-CT/amira/Ammonit-Eo_u-cropped-141x151x180-XY.vol.am}
set origVolume [load $volPath]

# Load OrthoSlice module and associate it to the original volume
set origOrthoSlice [create HxOrthoSlice]
$origOrthoSlice data connect $origVolume

# Set slice orientation as yz and automatically adjust view
$origOrthoSlice frameSettings setValue 0 0
$origOrthoSlice sliceOrientation setValue 2
$origOrthoSlice options setValue 0 1
$origOrthoSlice fire

# Take a snapshot of the orthoslice and hide it
$origOrthoSlice setViewerMask 1
viewer 0 redraw
viewer 0 snapshot -alpha /tmp/original.png
$origOrthoSlice setViewerMask 0

denoiseVolume $origVolume $origOrthoSlice 21 0.6 1

# Redraw viewer
viewer 0 setAutoRedraw 1
viewer 0 redraw
