import pytest
import tempfile
from pathlib import Path
from backend.project_manager import ProjectManager, get_default_project
from backend.models import SceneObject

def test_project_manager_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = ProjectManager(storage_dir=Path(tmpdir))
        orig_proj = pm.get_current()
        orig_name = orig_proj.name
        
        # Add an object
        new_obj = SceneObject(
            id="saved_box",
            name="Persistent Box",
            type="box",
            position=[2.0, 3.0, 0.4],
            size=[0.2, 0.2, 0.2]
        )
        orig_proj.objects.append(new_obj)
        
        # Save to disk
        saved_path = pm.save_project_to_disk("test_project.rfproj.json")
        assert Path(saved_path).exists()
        
        # Create a new manager and load
        pm2 = ProjectManager(storage_dir=Path(tmpdir))
        loaded = pm2.load_project_from_file("test_project.rfproj.json")
        assert loaded.name == orig_name
        assert any(o.id == "saved_box" for o in loaded.objects)

def test_project_manager_reset_to_saved():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = ProjectManager(storage_dir=Path(tmpdir))
        pm.save_project_to_disk("base.rfproj.json")
        
        # Mutate current project
        pm.get_current().name = "Modified Name"
        pm.get_current().objects.clear()
        
        # Reset to saved
        restored = pm.reset_to_saved()
        assert restored.name == "Workspace Alpha"
        assert len(restored.objects) > 0

def test_reject_incompatible_version():
    pm = ProjectManager()
    bad_json = '{"schema_version": "9.9.0", "name": "Future Project"}'
    with pytest.raises(ValueError, match="Unsupported schema version"):
        pm.load_project_from_json(bad_json)
