proc takeSnapshot {volName strOrientation} {
    global orthoSlice

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

    viewer 0 redraw
    viewer 0 snapshot -alpha /tmp/$volName-$strOrientation.png
}

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

# Hide any previous orthoslice in the pool
foreach orthoSlice [all HxOrthoSlice] {
    $orthoSlice setViewerMask 0
}

# Load OrthoSlice module and associate it to the original volume
set orthoSlice [create HxOrthoSlice]

# Take three snapshots of each uniform scalar field in the pool, each snapshot
# with a different orientation
foreach volume [all HxUniformScalarField3] {
    $orthoSlice data connect $volume

    takeSnapshot $volume "xy"
    takeSnapshot $volume "xz"
    takeSnapshot $volume "yz"
}


# Redraw viewer
viewer 0 setAutoRedraw 1
viewer 0 redraw
