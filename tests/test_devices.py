"""Hardware dimensions, mount axes and optional payload contracts."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from iphone_mujoco.devices import device_ids, get_device, phone_geometry


def _model(device_id, *, region="us", payload=False):
    asset, phone = phone_geometry(device_id, body_name="phone", region=region, payload=payload)
    root = ET.Element("mujoco")
    root.append(asset)
    world = ET.SubElement(root, "worldbody")
    # Exercise inherited geom defaults, including density and contact.
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", density="4500", contype="1", conaffinity="1")
    parent = ET.SubElement(world, "body", name="parent")
    ET.SubElement(parent, "joint", type="hinge")
    ET.SubElement(parent, "inertial", pos="0 0 0", mass="1", diaginertia=".01 .01 .01")
    parent.append(phone)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def test_catalog_copies_do_not_modify_other_readers():
    assert device_ids() == ("iphone-15-pro", "iphone-15-pro-max", "iphone-16-pro",
                            "iphone-16-pro-max", "iphone-17-pro", "iphone-17-pro-max")
    values = get_device("iphone-17-pro").to_dict()
    values["rear_lens_xy_mm"]["wide"][0] = 999
    assert get_device("iphone-17-pro").to_dict()["rear_lens_xy_mm"]["wide"][0] == 14.37
    with pytest.raises(ValueError, match="Unknown iPhone model"):
        get_device("iphone-17")


def test_sourced_dimensions_region_masses_and_camera_basis():
    pro, maximum = get_device("iphone-17-pro"), get_device("iphone-17-pro-max")
    np.testing.assert_allclose([pro.width_m, pro.height_m, pro.thickness_m], [.07185, .15001, .00875])
    np.testing.assert_allclose([maximum.width_m, maximum.height_m, maximum.thickness_m], [.07798, .16343, .00875])
    assert [pro.mass_kg("us"), pro.mass_kg("cn"), maximum.mass_kg("us"), maximum.mass_kg("cn")] == [.206, .204, .233, .231]
    # Rear-view drawing X points opposite to screen-right in the phone frame.
    pose = pro.phone_from_camera("wide")
    np.testing.assert_allclose(pose[:3, 3], [.035925 - .01437, .075005 - .03361, -.004375 - .00443])
    np.testing.assert_array_equal(pose[:3, :3], np.diag([1, -1, -1]))
    np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1)
    np.testing.assert_allclose(pro.phone_from_camera("ultra_wide")[:3, 3] - pose[:3, 3], [0, .01924, 0])
    np.testing.assert_allclose(pro.phone_from_camera("telephoto")[:3, 3] - pose[:3, 3], [-.01799, .00962, 0])
    with pytest.raises(ValueError, match="lens"):
        pro.phone_from_camera("front")
    with pytest.raises(ValueError, match="region"):
        pro.mass_kg("invalid")


def test_catalog_preserves_nominal_focal_lengths_and_crop_lenses():
    pro = get_device("iphone-17-pro").to_dict()
    modes = {mode["display_zoom"]: mode for mode in pro["zoom_modes"]}
    assert [(modes[zoom]["equivalent_focal_mm"], modes[zoom]["lens"], modes[zoom]["crop"])
            for zoom in [.5, 1, 2, 4, 8]] == [
                (13, "ultra_wide", False), (24, "wide", False), (48, "wide", True),
                (100, "telephoto", False), (200, "telephoto", True)]
    for identifier in device_ids():
        device_modes = {mode["display_zoom"]: mode for mode in get_device(identifier).to_dict()["zoom_modes"]}
        assert [(device_modes[zoom]["equivalent_focal_mm"], device_modes[zoom]["lens"],
                 device_modes[zoom]["crop"]) for zoom in (1.2, 1.5)] == [
                     (28, "wide", True), (35, "wide", True)]
    assert pro["geometry_status"]["camera_z"] == "lens_cover_surface_approximation"
    assert "Inferred" in pro["camera_identity_evidence"]
    assert get_device("iphone-15-pro").to_dict()["zoom_modes"][-1]["equivalent_focal_mm"] == 77
    assert get_device("iphone-16-pro").to_dict()["zoom_modes"][-1]["equivalent_focal_mm"] == 120


@pytest.mark.parametrize("device_id", device_ids())
def test_geometry_compiles_with_sourced_mesh_bounds_and_no_added_dynamics(device_id):
    device = get_device(device_id)
    asset, body = phone_geometry(device, body_name="phone")
    mesh = asset.find("mesh[@name='phone_shell_mesh']")
    vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
    np.testing.assert_allclose(np.ptp(vertices, axis=0),
                               [device.width_m, device.height_m, device.thickness_m])
    # At least one actual corner ordinate is retained, rather than a box mesh.
    assert len(vertices) > 8
    assert not np.any(np.all(np.isclose(np.abs(vertices[:, :2]),
                                        [device.width_m / 2, device.height_m / 2]), axis=1))
    assert body.find("inertial") is None
    model = _model(device_id)
    phone = model.body("phone").id
    assert model.body_mass[phone] == 0
    phone_geoms = np.flatnonzero(model.geom_bodyid == phone)
    np.testing.assert_array_equal(model.geom_contype[phone_geoms], 0)
    np.testing.assert_array_equal(model.geom_conaffinity[phone_geoms], 0)
    bounds = model.geom("phone_shell_bounds").id
    np.testing.assert_allclose(model.geom_size[bounds],
                              [device.width_m / 2, device.height_m / 2, device.thickness_m / 2])
    assert model.geom_rgba[bounds, 3] == 0
    for lens in ("wide", "ultra_wide", "telephoto"):
        cover = model.geom("phone_" + lens + "_lens").id
        np.testing.assert_allclose(model.geom_pos[cover, :2], device.phone_from_camera(lens)[:2, 3])
        np.testing.assert_allclose(model.geom_pos[cover, 2] - model.geom_size[cover, 1],
                                  device.phone_from_camera(lens)[2, 3])


def test_payload_uses_selected_nominal_mass_and_estimated_box_inertia():
    device = get_device("iphone-17-pro-max")
    model = _model(device.id, region="cn", payload=True)
    phone = model.body("phone").id
    mass = device.mass_kg("cn")
    assert model.body_mass[phone] == mass
    width, height, depth = device.width_m, device.height_m, device.thickness_m
    expected = mass / 12 * np.array([height**2 + depth**2, width**2 + depth**2, width**2 + height**2])
    np.testing.assert_allclose(model.body_inertia[phone], expected)
    np.testing.assert_array_equal(model.body_ipos[phone], 0)
    _, body = phone_geometry(device, payload=True)
    assert "uniform-box inertia estimate" in ET.tostring(body, encoding="unicode")


def test_geometry_rejects_invalid_group_payload_and_region():
    for kwargs in ({"visual_group": 6}, {"visual_group": True}, {"payload": "yes"},
                   {"region": "invalid"}, {"body_name": ""}):
        with pytest.raises(ValueError):
            phone_geometry("iphone-17-pro", **kwargs)
