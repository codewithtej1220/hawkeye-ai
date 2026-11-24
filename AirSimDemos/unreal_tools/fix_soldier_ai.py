"""Unreal editor script: make the soldier patrol.

The soldier's blueprint wanders with AI movement (startWander -> AIMoveTo), which needs
  1. an AI controller on the character (AIControllerClass was None, so nothing drove him), and
  2. a navigation mesh under him (the NavMeshBoundsVolume sat ~290 m away). It is kept small
     (30 m x 30 m) so he stays in one area the drone can watch.
Run with RUN_FIX_SOLDIER_AI.bat (Unreal must be closed). The map is saved in place."""
import unreal

MAP = "/Game/Maps/LandscapeMap"
PATROL_HALF_SIZE_UU = (1500.0, 1500.0, 2000.0)  # 30 m x 30 m patrol area (keeps him in view for demos), 40 m tall

unreal.EditorLoadingAndSavingUtils.load_map(MAP)
actors = unreal.EditorLevelLibrary.get_all_level_actors()
soldier = next(a for a in actors if a.get_name().startswith("ThirdPersonCharacter"))
volume = next(a for a in actors if a.get_class().get_name() == "NavMeshBoundsVolume")

soldier.set_editor_property("ai_controller_class", unreal.AIController.static_class())
soldier.set_editor_property("auto_possess_ai", unreal.AutoPossessAI.PLACED_IN_WORLD_OR_SPAWNED)

_, extent = volume.get_actor_bounds(False)
scale = volume.get_actor_scale3d()
volume.set_actor_location(soldier.get_actor_location(), False, False)
volume.set_actor_scale3d(unreal.Vector(scale.x * PATROL_HALF_SIZE_UU[0] / extent.x,
                                       scale.y * PATROL_HALF_SIZE_UU[1] / extent.y,
                                       scale.z * PATROL_HALF_SIZE_UU[2] / extent.z))

origin, extent = volume.get_actor_bounds(False)
unreal.log("HAWKEYE soldier at %s, AI controller %s" % (soldier.get_actor_location(), soldier.get_editor_property("ai_controller_class")))
unreal.log("HAWKEYE nav volume origin %s extent %s" % (origin, extent))
saved = unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
unreal.log("HAWKEYE saved: %s" % saved)
