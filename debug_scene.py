#!/usr/bin/env python3
"""
Debug script to inspect VIGA Blender scene content
"""
import bpy

def debug_scene():
    """Debug scene content."""
    print("🔍 Debugging VIGA scene content...")
    print(f"📦 Scene name: {bpy.context.scene.name}")
    print(f"📊 Total objects: {len(bpy.context.scene.objects)}")
    
    for obj in bpy.context.scene.objects:
        print(f"  • {obj.name} (Type: {obj.type}, Visible: {obj.visible_get()})")
        if hasattr(obj, 'children') and obj.children:
            for child in obj.children:
                print(f"    ↳ {child.name} (Type: {child.type})")
    
    # Check collections
    print(f"\n📁 Collections:")
    for collection in bpy.data.collections:
        print(f"  • {collection.name} ({len(collection.objects)} objects)")
        for obj in collection.objects:
            print(f"    ↳ {obj.name} (Type: {obj.type})")

if __name__ == "__main__":
    debug_scene()