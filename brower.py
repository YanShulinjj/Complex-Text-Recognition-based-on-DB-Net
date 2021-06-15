import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import QWebEngineView


class MyBrowser(QMainWindow):
    def __init__(self, keyword, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('My Browser')
        self.showMaximized()

        self.webview = WebEngineView()
        self.webview.load(QUrl("https://s.taobao.com/search?q={}".format(keyword)))
        self.setCentralWidget(self.webview)


################################################
#######创建浏览器
################################################
class WebEngineView(QWebEngineView):
    windowList = []

    # 重写createwindow()
    def createWindow(self, QWebEnginePage_WebWindowType):
        new_webview = WebEngineView()
        new_window = MyBrowser()
        new_window.setCentralWidget(new_webview)
        # new_window.show()
        self.windowList.append(new_window)  # 注：没有这句会崩溃！！！
        return new_webview
