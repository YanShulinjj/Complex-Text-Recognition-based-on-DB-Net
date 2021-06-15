import os
import re
import requests
keys = [
    '600可口可乐',
    '600百事可乐',
    'rio鸡尾酒',
    '奥妙洗衣液',
    '清扬洗发水',
    '加多宝瓶装',
    '江小白酒图片',
    '娇子香烟图片',
    '康师傅纯净水图片',
    '蓝月亮图片',
    '老白干图片',
    '六个核桃图片',
    '飘柔图片',
    '哇哈哈ad钙奶图片',
    '黑人牙膏图片',
    '戴尔显示器图片'
]
headers = {
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding':'gzip, deflate, br',
    'Accept-Language':'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.128 Safari/537.36 Edg/89.0.774.77',
    'Referer':'https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=%E5%8F%AF%E4%B9%90',
    #参考链接:https://image.baidu.com/
    #请在此处填写你的 Cookie
}

def dowmloadPic(html, keyword, n):
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)
    i = 1
    print('找到关键词:' + keyword + '的图片，现在开始下载图片...')
    for each in pic_url:
        print('正在下载第' + str(i) + '张图片，图片地址:' + str(each))
        try:
            pic = requests.get(each, timeout=10)
        except requests.exceptions.ConnectionError:
            print('【错误】当前图片无法下载')
            continue

        dir = './images/test/' + str(n)
        if not os.path.exists(dir):
            os.mkdir(dir)
        fname = dir + '/' + str(i) + '.jpg'
        fp = open(fname, 'wb')
        fp.write(pic.content)
        fp.close()
        i += 1

def run():
    for i, word in enumerate(keys):
        url = 'https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=' + word
        result = requests.get(url, headers=headers)
        # print(result.text)
        dowmloadPic(result.text, word, i)




if __name__ == '__main__':
    run()