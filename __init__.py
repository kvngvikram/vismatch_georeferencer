def classFactory(iface):
    from .plugin import VisMatchGeorefPlugin
    return VisMatchGeorefPlugin(iface)
