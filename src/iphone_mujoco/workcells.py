"""Dimensioned inspection fixtures and neutral lighting for arm experiments."""
from __future__ import annotations

import xml.etree.ElementTree as ET


def add_arm_workcell(root: ET.Element) -> None:
    """Add a bolted workbench and two machined inspection targets.

    Robot geometry, joints and inertias remain in the upstream model. These
    static fixtures provide measurable surfaces for wrist RGB-D experiments.
    """
    world = root.find("worldbody")
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", name="cell_sky", type="skybox", builtin="gradient",
                  rgb1=".60 .65 .69", rgb2=".85 .88 .90", width="512", height="3072")
    for name, color, specular in (
        ("cell_table", ".48 .51 .54 1", ".35"),
        ("cell_frame", ".23 .27 .31 1", ".6"),
        ("cell_alloy", ".68 .70 .72 1", ".8"),
        ("cell_dark", ".07 .09 .11 1", ".2"),
    ):
        ET.SubElement(assets, "material", name=name, rgba=color,
                      specular=specular, shininess=".45")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", ".28 .28 .28")
    headlight.set("diffuse", ".55 .55 .55")
    if visual.find("quality") is None:
        ET.SubElement(visual, "quality", shadowsize="4096")
    ET.SubElement(world, "light", name="cell_key", pos="0 -2 3", dir=".1 .3 -1",
                  diffuse=".8 .8 .8", specular=".4 .4 .4")
    ET.SubElement(world, "light", name="cell_fill", pos="1 2 2", dir="-.3 -.3 -1",
                  diffuse=".4 .4 .4", castshadow="false")

    def geom(name, kind, pos, size, material="cell_alloy", **kwargs):
        return ET.SubElement(world, "geom", name=name, type=kind, pos=pos,
                             size=size, material=material, **kwargs)

    geom("cell_floor", "plane", "0 0 -.81", "5 5 .1", "cell_frame")
    geom("cell_tabletop", "box", ".12 0 -.035", ".80 .72 .035", "cell_table")
    for x in (-.59, .83):
        for y in (-.63, .63):
            key = f"{x}_{y}"
            geom("cell_leg_" + key, "box", f"{x} {y} -.43", ".025 .025 .36", "cell_frame")
            geom("cell_foot_" + key, "cylinder", f"{x} {y} -.79", ".035 .015", "cell_dark")
    for y in (-.63, .63):
        geom("cell_crossbar_" + str(y), "box", f".12 {y} -.62", ".72 .02 .02", "cell_frame")
    geom("robot_mount_plate", "cylinder", "0 0 .004", ".12 .004", "cell_alloy")
    for x in (-.085, .085):
        for y in (-.055, .055):
            geom(f"robot_mount_bolt_{x}_{y}", "cylinder", f"{x} {y} .010", ".006 .003", "cell_dark")
    # Target origins are moved together beneath each configured wrist camera.
    geom("gauge_block", "box", ".5 .3 .05", ".05 .05 .05", "cell_alloy")
    geom("gauge_block_slot", "box", ".5 .3 .1001", ".026 .006 .0001", "cell_dark",
         contype="0", conaffinity="0")
    geom("gauge_block_key", "box", ".47 .3 .1002", ".004 .035 .0002", "cell_frame",
         contype="0", conaffinity="0")
    geom("inspection_boss", "cylinder", ".4 -.25 .05", ".05 .05", "cell_alloy")
    geom("inspection_boss_bore", "cylinder", ".4 -.25 .1002", ".022 .0002", "cell_dark",
         contype="0", conaffinity="0")
    geom("inspection_boss_rim", "cylinder", ".4 -.25 .102", ".035 .002", "cell_frame",
         contype="0", conaffinity="0")
