import gradio as gr
import numpy as np
import os, json, re, shutil, math

from dokithemejupyter.themes import themes
from PIL import ImageColor, Image, ImageCms

self_dir = os.path.dirname(os.path.realpath(__file__))

BG_SAVE_KEY = "doki_background"
DEFAULT_CONFIG_LOCATION = os.path.join(self_dir, "doki_settings.json")
DEFAULT_TEMPLATE_LOCATION = os.path.join(self_dir, "templates")
DEFAULT_OPACITY = 0.08

# https://github.com/ruozhichen/rgb2Lab-rgb2hsl/blob/master/LAB.py
def rgb2lab(inputColor):
	RGB=[0,0,0]
	for i in range(0,len(inputColor)):
		RGB[i]=inputColor[i]/255.0

	X=RGB[0]*0.4124+RGB[1]*0.3576+RGB[2]*0.1805
	Y=RGB[0]*0.2126+RGB[1]*0.7152+RGB[2]*0.0722
	Z=RGB[0]*0.0193+RGB[1]*0.1192+RGB[2]*0.9505
	XYZ=[X,Y,Z]
	XYZ[0]/=95.045/100
	XYZ[1]/=100.0/100
	XYZ[2]/=108.875/100

	L=0
	for i in range(0,3):
		v=XYZ[i]
		if v>0.008856:
			v=pow(v,1.0/3)
			if i==1:
				L=116.0*v-16.0
		else:
			v*=7.787
			v+=16.0/116
			if i==1:
				L=903.3*XYZ[i]
		XYZ[i]=v

	a=500.0*(XYZ[0]-XYZ[1])
	b=200.0*(XYZ[1]-XYZ[2])
	return [L, a, b]

def lab2rgb(inputColor):
	L, a, b = inputColor
	T1 = 0.008856
	T2 = 0.206893
	d = T2
	fy =math.pow( (L + 16) / 116.0,3)
	fx = fy + a / 500.0
	fz = fy - b / 200.0
	fy = (fy) if (fy > T1) else ( L/903.3)
	Y=fy
	fy=(math.pow(fy,1.0/3)) if (fy > T1) else (7.787*fy+16.0/116)

	# compute original XYZ[0]
	fx=fy+a/500.0
	X=(math.pow(fx,3.0)) if (fx > T2) else ((fx-16.0/116)/7.787)

	# compute original XYZ[2]
	fz=fy-b/200.0
	Z=(math.pow(fz,3.0)) if (fz >T2) else ((fz-16.0/116)/7.787)

	X *= 0.95045
	Z *= 1.08875
	R = 3.240479 * X + (-1.537150) * Y + (-0.498535) * Z
	G = (-0.969256) * X + 1.875992 * Y + 0.041556 * Z
	B = 0.055648 * X + (-0.204043) * Y + 1.057311 * Z

	RGB = [R, G, B]
	for i in range(0,3):
		RGB[i] = min(int(round(RGB[i] * 255)),255)
		RGB[i] = max(RGB[i],0)
	return RGB

def _cv2_make_transparent_and_save(file, outpath, outkey, opacity):
    """
    @see make_transparent_and_save
    """
    import cv2
    image_data = cv2.imread(file, cv2.IMREAD_COLOR)
    image_rgba = cv2.cvtColor(image_data, cv2.COLOR_RGB2RGBA)
    image_rgba[..., 3] = (image_rgba[..., 3] * opacity).astype(np.uint8)
    
    outname = os.path.join(outpath, outkey + ".webp")
    cv2.imwrite(outname, image_rgba, [cv2.IMWRITE_WEBP_QUALITY, 95])
    return outname

def make_transparent_and_save(file, outpath, outkey, opacity):
    """
    Makes an image transparent and save it as WebP if it was supported
    (needs libwebp or w/e it is called) or fallback save as PNG.
    Uses CV2 if found, PIL otherwise.

    @param file {str} Image file path.
    @param outpath {str} Output directory.
    @param outkey {str} Output filename without extension.
    @param opacity {float} Image opacity.
    @returns {str} Output file path.
    """
    try:
        import cv2
        return _cv2_make_transparent_and_save(file, outpath, outkey, opacity)
    except (ModuleNotFoundError, ImportError):
        im = Image.open(file).convert("RGBA")
        
        im2 = im.copy()
        im2.putalpha(round(255 * opacity))
        im.paste(im2, im)

        try:
            outname = os.path.join(outpath, outkey + ".webp")
            im.save(outname, "WEBP", quality=95)
            return outname
        except KeyError: # webp support not installed
            outname = os.path.join(outpath, outkey + ".png")
            im.save(outname, "PNG", optimize=True)
            return outname

def to_rgba(color, opacity):
    """
    Converts color to CSS RGBA format, adding some transparency.

    @param color {str} CSS color of any kind.
    @param opacity {float} Target opacity of the color.
    @returns {str} CSS color of RGBA format.
    """
    rgb = ImageColor.getcolor(color, "RGB")
    return "rgba({}, {}, {}, {})".format(*rgb, opacity)

def change_lumi(color, delta):
    """
    Modifies the luminance of a color by converting it to LAB and back.

    @param color {str} CSS color of any kind.
    @param delta {int} Increment of luminance, range from -255 to 255.
    @returns {str} CSS color of RGB format.
    """
    if isinstance(color, str):
        rgb = ImageColor.getcolor(color, "RGB")
    else:
        rgb = color
        
    lab = rgb2lab(rgb)
    lab[0] = min(100, max(0, lab[0] + delta * 100.0 / 255.0))
    rgb = lab2rgb(lab)
    
    return "rgb({}, {}, {})".format(*rgb)

def generate_gradio_css(tempura, theme_color):
    """
    Generates the CSS declaration of the Doki Theme for the Gradio app.

    @param tempura {str} Format string containing parameters nicely contained in curly braces.
    @param theme_color {dict} Dictionary containing every color using Doki definition.
    @returns {str} Expressively generated CSS stylesheet.
    """
    css = tempura.format(
        shadowColor = theme_color["selectionBackground"],
        shadowColorTransparent = to_rgba(theme_color["selectionBackground"], 0.7),
        accentShadowColor = to_rgba(theme_color["accentColor"], 0.7),
        accentColorReallyTransparent = to_rgba(theme_color["accentColor"], 0.3),
        accentColorDeeper = change_lumi(theme_color["accentColor"], -25),
        gradientFrom = theme_color["baseBackground"],
        gradientTo = change_lumi(theme_color["baseBackground"], 25),
        secondaryGradientFrom = theme_color["secondaryBackground"],
        secondaryGradientTo = change_lumi(theme_color["highlightColor"], 25),
        primaryGradientFrom = change_lumi(theme_color["selectionBackground"], -25),
        primaryGradientTo = theme_color["selectionBackground"],
        primaryHover = theme_color["accentColor"],
        highlightShadowColor = theme_color["foldedTextBackground"],
        errorGradientFrom = to_rgba(theme_color["errorColor"], 0.2),
        errorGradientTo = to_rgba(theme_color["errorColor"], 0.3),
        **theme_color
    )
    return css

def save_settings(theme_select, file_bg, bg_align, bg_opacity, files_dir = "files", config_file = DEFAULT_CONFIG_LOCATION):
    """
    Saves the settings to a local file (doki_settings.json).

    @param theme_select {str} Theme selected. Should be within the keys of dokithemejupyter.themes.themes.
    @param file_bg {tempfile._TemporaryFileWrapper} The temporary file uploaded to Web UI backend.
    @param bg_align {str} Alignment of background, e.g. chaos neutral or lawful good.
    (For real, the CSS background position property)
    @param bg_opacity {float} Opacity of the background. A value of 0 turns off the background.
    @param files_dir {str} The directory that points to the files folder gradio serves
    (such that its files can be accessed via http://server/files=files/<filename>)
    @param config_file {str} Config file location.
    @returns {str} Result string to display in the UI.
    """
    try:
        if not os.path.isdir(files_dir):
            os.makedirs(files_dir)
            
        settings = {
            "theme" : "",
            "bg" : "",
            "bg_align" : "center" if bg_align is None else bg_align,
            "bg_opacity" : float(bg_opacity)
        }
        
        if theme_select in themes:
            settings["theme"] = theme_select
        else:
            settings["theme"] = ""

        if file_bg is None:
            if theme_select in themes and bg_opacity >= 0.001:
                settings["bg"] = "https://doki.assets.unthrottled.io/backgrounds/wallpapers/transparent/{}".format(
                    themes[theme_select]["colors"]["stickerName"]
                )
                # Set opacity back to default value if bg is not present
                settings["bg_opacity"] = DEFAULT_OPACITY
            else:
                settings["bg"] = ""
                settings["bg_opacity"] = DEFAULT_OPACITY
        elif bg_opacity < 0.001:
            settings["bg"] = ""
            settings["bg_opacity"] = DEFAULT_OPACITY
        elif bg_opacity > 0.999:
            ext = os.path.splitext(file_bg.name)[1]
            file_saved = os.path.join(files_dir, "{}{}".format(BG_SAVE_KEY, ext))
            shutil.copy(file_bg.name, file_saved)
            settings["bg"] = "/file=files/{}".format(
                os.path.basename(file_saved)
            )
        else:
            file_saved = make_transparent_and_save(
                file_bg.name,
                files_dir,
                BG_SAVE_KEY,
                settings["bg_opacity"]
            )
            settings["bg"] = "/file=files/{}".format(
                os.path.basename(file_saved)
            )
            
        with open(config_file, "w", encoding = "utf-8") as fp:
            json.dump(settings, fp)

        return "Settings updated: {}\nRestart Gradio to see effects!".format(json.dumps(settings))

    except Exception as ex:
        return "Error: {}".format(ex)

def get_settings(config_file = DEFAULT_CONFIG_LOCATION):
    try:
        with open(config_file, encoding = "utf-8") as fp:
            settings = json.load(fp)
    except:
        settings = {
            "theme" : "",
            "bg" : "",
            "bg_align" : "center",
            "bg_opacity" : DEFAULT_OPACITY
        }

    return settings

def theme(template_dir = DEFAULT_TEMPLATE_LOCATION, config_file = DEFAULT_CONFIG_LOCATION):
    """
    Adds Doki style to the Gradio app.
    """
    # load settings
    with open(os.path.join(template_dir, "tempura.css"), encoding = "utf-8") as fp:
        css_template = fp.read()

    with open(os.path.join(template_dir, "background.css"), encoding = "utf-8") as fp:
        css_template_bg = fp.read()

    settings = get_settings(config_file = config_file)

    # bg
    if len(settings["bg"]) <= 0 or settings["bg"].lower() == "none":
        css_bg = ""
    else:
        if settings["bg_align"] == "random":
            # the chaos evil option
            alignment = "{}% {}%".format(
                np.random.random() * 100,
                np.random.random() * 100
            )
        else:
            alignment = settings["bg_align"]
        css_bg = css_template_bg.format(
            img = settings["bg"],
            align = alignment,
            opacity = settings["bg_opacity"]
        )

    # theme
    theme = settings["theme"]
    if theme not in themes:
        css = ""
    else:
        css = generate_gradio_css(css_template, themes[theme]['colors'])

    return gr.HTML(f"""
        <style>{css} {css_bg}</style>
    """)

def theme_settings(files_dir = "files", config_file = DEFAULT_CONFIG_LOCATION):
    """
    Adds a block in the UI to adjust the settings.
    """
    settings = get_settings(config_file = config_file)
    
    with gr.Blocks(analytics_enabled=False) as ui:
        theme_select = gr.Dropdown(label="Theme", value=settings["theme"], choices = list(themes.keys()))
        file_bg = gr.File(label="Background Image")
        bg_align = gr.Dropdown(label="Background Align", value=settings["bg_align"], choices=[
            "left",
            "center",
            "right",
            "center -5%",
            "center -10%",
            "center -15%",
            "random",
        ])
        bg_opacity = gr.Slider(0, 1, value=settings["bg_opacity"], label="Background Opacity")
        update = gr.Button(value="Save Settings", variant='primary')

        result = gr.Text(value="", show_label=False)

        update.click(
            fn=save_settings,
            inputs=[
                theme_select,
                file_bg,
                bg_align,
                bg_opacity
            ],
            outputs=result
        )

    return ui
