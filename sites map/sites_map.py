import base64
import folium
import pandas as pd
import branca
map = folium.Map(location = [32.05822280781782, 35.30622640955295], zoom_start=10, tiles="Stamen Terrain",control_scale=True)



#load the data- volacons in the US
data = pd.read_excel("./site_coridinate.xlsx")
lat = list(data.lat)
lon = list(data.lon)
names = list(data.Sites)


# def color_maker(elevation):
#     if elevation < 1000:
#         return 'green'
#     elif 1000<=elevation <3000:
#         return 'orange'

#     return 'red'

legend_html = '''
{% macro html(this, kwargs) %}
<div style="
    position: fixed; 
    bottom: 126px;
    width: 229px;
    height: 50px;
    z-index:9999;
    font-size:14px;
    ">
    <p><img src = "tent.png" style="width:20px;height:20px;margin-left:20px"> Camping</p>
    <p><img src = "pin.png" style="width:20px;height:20px;margin-left:20px"> Lookout</p>
    <p><img src = "drop.png" style="width:20px;height:20px;margin-left:20px"> Water Source</p>
    <p><img src = "archeology.png" style="width:20px;height:20px;margin-left:20px"> Archeology</p>

</div>
<div style="
    position: fixed; 
    bottom: 45px;
    left: 8px;
    width: 129px;
    height: 150px; 
    z-index:9998;
    font-size:14px;
    background-color: #ffffff;

    opacity: 0.7;
    ">
</div>
{% endmacro %}'''
legend = branca.element.MacroElement()
legend._template = branca.element.Template(legend_html)

html = """ Volcano name: <br> <a href = "https://www.google.com/search?q=%%22%s%%22" target=_blank>%s</a><br>
       Height: %s m"""

tent_encoded = base64.b64encode(open('tent_resized.png', 'rb').read()).decode()
archeology_encoded = base64.b64encode(open('archeology_resized.png', 'rb').read()).decode()
# paw_encoded = base64.b64encode(open('paw_resized.png', 'rb').read()).decode()
drop_encoded = base64.b64encode(open('drop_resized.png', 'rb').read()).decode()
pin_encoded = base64.b64encode(open('pin_resized.png', 'rb').read()).decode()



masada_html = f''' 
<p style="text-align:center;font-size:29px;margin:5px">Masada</p>
<div style = "text-align:center;">
    <img src="data:image/png;base64,{tent_encoded}"  width="35" height="35">
    <img src="data:image/png;base64,{archeology_encoded}"  width="35" height="35">
    
    </div>

'''
caesarea_html = f""" 
<p style = "text-align:center;font-size:29px;margin:5px">Caesarea</p>
<div style = "text-align:center">
    <img src="data:image/png;base64,{archeology_encoded}" width="35" height="35">
    </div>
"""
gedi_html = f""" 
<div>
<p style = "text-align:center;font-size:29px;margin:5px">En Gedi</p>

<div style = "text-align:center">
    <img src="data:image/png;base64,{archeology_encoded}"  width="35" height="35">
    <img src="data:image/png;base64,{drop_encoded}"  width="35" height="35">
</div>
</div>
"""

hermon_html = f""" 
<p style = "text-align:center;font-size:29px;margin:5px">Hermon Stream (Banias)</p>

    <div style = "text-align:center">
    <img src="data:image/png;base64,{archeology_encoded}"  width="35" height="35">
    <img src="data:image/png;base64,{drop_encoded}"  width="35" height="35">
    </div>
"""

#we have to groups that will contain different layers- one for the volcanoes and another for the population

feature_group_sites = folium.FeatureGroup(name = "Sites")

for lat,lon,name in zip(lat,lon,names):
    chosen_sites = ['The Masada','En Gedi','Hermon Stream (Banias)','Caesarea']
    #add object to the Map
    if name in chosen_sites:
        if name == 'The Masada':
            html = masada_html
        elif name == 'Caesarea':
            html = caesarea_html
        elif name == 'En Gedi':
            html = gedi_html
        elif name == 'Hermon Stream (Banias)':
            html = hermon_html

        # if name == 'En Gedi':
        #     iframe = folium.IFrame(html, width=190, height=65)
        # else : iframe = folium.IFrame(html, width=190, height=70)
        # popup = folium.Popup(iframe , show = True,max_width=150,min_width = 150)
        # marker = folium.Marker([lat,lon], popup=(popup)).add_to(map)
        # map.add_child(marker)

        # folium.Marker(
        #     location = [lat,lon],
        #     popup = folium.Popup(html, parse_html=False, max_width=1000,show=True),
        #     icon = folium.Icon(prefix = 'fa')
        #     ).add_to(map)

        # iframe = folium.IFrame(html)
        # popup = folium.Popup(iframe,
        #                     min_width=100,
        #                     max_width=50, show=True)


        # marker = folium.Marker([lat, lon],
        #                     popup=popup).add_to(map)


        popup = folium.Popup(folium.Html(html, script=True,width='100%',height='100%'), max_width=500,show = True)
        #iframe = folium.IFrame(html= bla % (name),width=180, height=150) #this will be the popup
        feature_group_sites.add_child(folium.Marker(location= [lat,lon], radius = 6, popup= popup,color='grey',fill_opacity=0.7))


# from folium.plugins import FloatImage
# image_file = 'tent.PNG'

# FloatImage(image_file, bottom=0, left=0).add_to(map)
map.add_child(feature_group_sites)

#with LayerControl we can select different layers and dactivate them
map.add_child(folium.LayerControl())
map.get_root().add_child(legend)
map.save("Sites_Map.html")