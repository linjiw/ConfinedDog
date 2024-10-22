from pxr import Usd, UsdGeom

# Open the USD file
stage = Usd.Stage.Open('f1_car.usd')

# Get the root layer
root_layer = stage.GetRootLayer()

# Print basic file info
print(f"File path: {root_layer.realPath}")
print(f"Up axis: {UsdGeom.GetStageUpAxis(stage)}")

# Iterate through all prims in the stage
for prim in stage.Traverse():
    print(f"\nPrim path: {prim.GetPath()}")
    print(f"Prim type: {prim.GetTypeName()}")
    
    # Print attributes
    for attribute in prim.GetAttributes():
        print(f"  Attribute: {attribute.GetName()} = {attribute.Get()}")