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

# Generate a copy of the original volume and orthoslice and hide the orthoslice
set currVolume [$origVolume duplicate]
set currOrthoSlice [$origOrthoSlice duplicate]

$currOrthoSlice data connect $currVolume
$currOrthoSlice frameSettings setValue 0 0
$currOrthoSlice sliceOrientation setValue 2
$currOrthoSlice options setValue 0 1
$currOrthoSlice fire

# Open an editor for the current volume
$currVolume setEditor [create HxImageVisionEditor]
set currFilter [$currVolume getEditor]

# Go to the NLM denoising filter
$currFilter filter setValue 14
$currFilter fire

# Change the parameters of the filter
$currFilter searchWindow setValue 21
$currFilter similarityValue setValue 0.7

# Start filtering
$currFilter doIt hit
$currFilter fire

# Change name of the volume
set currVolName {window-21-similarity-07-iteration-1}
$currVolume setLabel $currVolName

# Take a snapshot of the result and hide the volume and the orthoslice
$currOrthoSlice setViewerMask 1
viewer 0 redraw
viewer 0 snapshot -alpha /tmp/$currVolName.png
$currOrthoSlice setViewerMask 0

# # Redraw viewer
viewer 0 setAutoRedraw 1
viewer 0 redraw
