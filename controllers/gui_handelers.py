from controllers.image_handelers import handle_upload_image
from controllers.plot_updates import (
    invert_contrast,
    update_plot_range,
    update_ylim_1d,
    zoom_2d_ctf,
    zoom_2d_image,
    adjust_contrast_fft,
    adjust_contrast_image,
    update_grayness,
    update_display_1d,
    update_tomo,
)


def setup_event_handlers(ctrl):
    """
    Connect signals from PyQt widgets to appropriate callbacks
    to update or reset the CTF parameters and refresh the plots.
    """
    ctrl.ui.voltage_slider.valueChanged.connect(
        lambda value, key="voltage": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.voltage_stability_slider.valueChanged.connect(
        lambda value, key="voltage_stability": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.electron_source_angle_slider.valueChanged.connect(
        lambda value, key="es_angle": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.electron_source_spread_slider.valueChanged.connect(
        lambda value, key="es_spread": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.chromatic_aberr_slider.valueChanged.connect(
        lambda value, key="cc": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.spherical_aberr_slider.valueChanged.connect(
        lambda value, key="cs": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.obj_lens_stability_slider.valueChanged.connect(
        lambda value, key="obj_stability": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.detector_combo.currentIndexChanged.connect(
        lambda value, key="detector": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.pixel_size_slider.valueChanged.connect(
        lambda value, key="pixel_size": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.sample_size_tomo.valueChanged.connect(
        lambda value, key="sample_size": update_tomo(ctrl, key, value)
    )
    ctrl.ui.defocus_slider.valueChanged.connect(
        lambda value, key="df": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.defocus_diff_slider_2d.valueChanged.connect(
        lambda value, key="df_diff": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.defocus_diff_slider_ice.valueChanged.connect(
        lambda value, key="df_diff": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.defocus_diff_slider_image.valueChanged.connect(
        lambda value, key="df_diff": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.defocus_az_slider_2d.valueChanged.connect(
        lambda value, key="df_az": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.defocus_az_slider_ice.valueChanged.connect(
        lambda value, key="df_az": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.defocus_az_slider_image.valueChanged.connect(
        lambda value, key="df_az": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.amplitude_contrast_slider.valueChanged.connect(
        lambda value, key="ac": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.additional_phase_slider.valueChanged.connect(
        lambda value, key="phase": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.plot_1d_x_min.valueChanged.connect(lambda _: update_plot_range(ctrl))
    ctrl.ui.plot_1d_x_max.valueChanged.connect(lambda _: update_plot_range(ctrl))
    ctrl.ui.plot_1d_y_min.valueChanged.connect(lambda _: update_plot_range(ctrl))
    ctrl.ui.plot_1d_y_max.valueChanged.connect(lambda _: update_plot_range(ctrl))
    ctrl.ui.plot_2d_x_min.valueChanged.connect(lambda _: update_plot_range(ctrl))
    ctrl.ui.plot_2d_x_max.valueChanged.connect(lambda _: update_plot_range(ctrl))
    ctrl.ui.plot_2d_y_min.valueChanged.connect(lambda _: update_plot_range(ctrl))
    ctrl.ui.plot_2d_y_max.valueChanged.connect(lambda _: update_plot_range(ctrl))
    ctrl.ui.freq_scale_2d.valueChanged.connect(lambda _: zoom_2d_ctf(ctrl))
    ctrl.ui.xlim_slider_1d.valueChanged.connect(lambda value: ctrl.ui.plot_1d_x_max.setValue(value))
    ctrl.ui.ylim_slider_1d.valueChanged.connect(lambda value: update_ylim_1d(ctrl, value))
    ctrl.ui.xlim_slider_ice.valueChanged.connect(lambda _: update_plot_range(ctrl))
    ctrl.ui.freq_scale_ice.valueChanged.connect(lambda _: zoom_2d_ctf(ctrl))
    ctrl.ui.gray_scale_2d.valueChanged.connect(lambda _: update_grayness(ctrl))
    ctrl.ui.gray_scale_ice.valueChanged.connect(lambda _: update_grayness(ctrl))
    ctrl.ui.gray_scale_tomo.valueChanged.connect(lambda _: update_grayness(ctrl))
    ctrl.ui.temporal_env_check.stateChanged.connect(
        lambda value, key="temporal_env": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.spatial_env_check.stateChanged.connect(
        lambda value, key="spatial_env": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.detector_env_check.stateChanged.connect(
        lambda value, key="detector_env": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.show_temp.stateChanged.connect(lambda _: update_display_1d(ctrl))
    ctrl.ui.show_spatial.stateChanged.connect(lambda _: update_display_1d(ctrl))
    ctrl.ui.show_detector.stateChanged.connect(lambda _: update_display_1d(ctrl))
    ctrl.ui.show_total.stateChanged.connect(lambda _: update_display_1d(ctrl))
    ctrl.ui.show_y0.stateChanged.connect(lambda _: update_display_1d(ctrl))
    ctrl.ui.show_legend.stateChanged.connect(lambda _: update_display_1d(ctrl))
    ctrl.ui.plot_tabs.currentChanged.connect(ctrl.on_tab_switched)
    ctrl.ui.reset_button.clicked.connect(ctrl.reset_parameters)
    ctrl.ui.save_img_button.clicked.connect(ctrl.save_plot)
    ctrl.ui.save_csv_button.clicked.connect(ctrl.save_csv)
    ctrl.ui.canvas_1d.mpl_connect("motion_notify_event", ctrl.on_hover)
    ctrl.ui.canvas_2d.mpl_connect("motion_notify_event", ctrl.on_hover)
    ctrl.ui.canvas_ice.mpl_connect("motion_notify_event", ctrl.on_hover)
    ctrl.ui.canvas_tomo.mpl_connect("motion_notify_event", ctrl.on_hover)
    ctrl.ui.canvas_image.mpl_connect("motion_notify_event", ctrl.on_hover)
    ctrl.ui.ice_thickness_slider.valueChanged.connect(
        lambda value, key="ice": ctrl.on_ctf_param_changed(key, value)
    )
    ctrl.ui.radio_button_group.buttonToggled.connect(ctrl.update_wrap_func)
    ctrl.ui.tilt_slider_tomo.valueChanged.connect(
        lambda value, key="tilt_angle": update_tomo(ctrl, key, value)
    )
    ctrl.ui.sample_thickness_slider_tomo.valueChanged.connect(
        lambda value, key="thickness": update_tomo(ctrl, key, value)
    )
    ctrl.ui.defocus_diff_slider_tomo.valueChanged.connect(
        lambda value, key="df_diff": update_tomo(ctrl, key, value)
    )
    ctrl.ui.defocus_az_slider_tomo.valueChanged.connect(
        lambda value, key="df_az": update_tomo(ctrl, key, value)
    )
    ctrl.ui.upload_btn.clicked.connect(lambda _: handle_upload_image(ctrl))
    ctrl.ui.invert_btn.clicked.connect(lambda _: invert_contrast(ctrl))
    ctrl.ui.size_scale_image.valueChanged.connect(
        lambda value, key="image": zoom_2d_image(ctrl, key, value)
    )
    ctrl.ui.size_scale_fft.valueChanged.connect(
        lambda value, key="fft": zoom_2d_image(ctrl, key, value)
    )
    ctrl.ui.contrast_scale_image.valueChanged.connect(lambda _: adjust_contrast_image(ctrl))
    ctrl.ui.contrast_scale_fft.valueChanged.connect(lambda value: adjust_contrast_fft(ctrl, value))
    ctrl.ui.info_button_1d.clicked.connect(ctrl.show_info)
    ctrl.ui.info_button_2d.clicked.connect(ctrl.show_info)
    ctrl.ui.info_button_ice.clicked.connect(ctrl.show_info)
    ctrl.ui.info_button_tomo.clicked.connect(ctrl.show_info)
    ctrl.ui.info_button_image.clicked.connect(ctrl.show_info)
    for button in ctrl.annotation_toggle_buttons:
        button.clicked.connect(ctrl.handle_annotation_toggle)
    ctrl.ui.contrast_sync_checkbox.stateChanged.connect(lambda _: adjust_contrast_image(ctrl))
