# 待安装环境pandoc~~操作本质是os.system调用命令行执行该程序
# pip install pywin32
# 调用了word环境
import os
import shutil
from win32com.client import constants, gencache


# markdown文件转换模块  markdown——>docx
def path(dir_path, func):
    """
    :param dir_path: 目标转换文件夹路径
    :param func: 功能 数字1：将markdown转换为doc  数字2：将doc转换为pdf 数字3： 将生成的pdf文件，移动到D：/filestorage目录下
    :return:
    """
    # os.walk输出当前目录下所有文件
    # x路径名， y文件夹名， z文件名
    for i in range(func):
        for x, y, z in os.walk(dir_path):
            for filename in z:
                # 分割文件名和后缀名
                suffix_name = os.path.splitext(filename)[1]
                prefix_name = os.path.splitext(filename)[0]
                filepath = x + '\\' + filename
                outfilepath = x + '\\' + prefix_name
                if i == 0:
                    if suffix_name == '.md':
                        print('pandoc ' + '"' + filepath + '"' + ' -o ' + x + '\\' + os.path.splitext(filename)[0] + '.docx')
                        os.system('pandoc ' + '"' + filepath + '"' + ' -o ' + '"' + x + '\\' + os.path.splitext(filename)[
                            0] + '.docx' + '"')
                elif i == 1:
                    if suffix_name == '.docx':
                        createPdf(filepath, outfilepath)
                        os.remove(filepath)
                elif i == 2:
                    if suffix_name == '.pdf':
                        # 不存在该目录，则利用mkdir进行创建
                        if not os.path.exists('D:\\filestorage\\'):
                            os.mkdir('D:\\filestorage\\')
                        shutil.move(filepath, 'D:\\filestorage\\'+prefix_name+'.pdf')


# docx文件转换模块 docx——>pdf
def createPdf(wordpath, pdfpath):
    """
    :param wordpath: 目标word路径
    :param pdfpath: 生成pdf路径
    :return:
    """
    word = gencache.EnsureDispatch('Word.Application')
    doc = word.Documents.Open(wordpath, ReadOnly=1)
    doc.ExportAsFixedFormat(pdfpath,
                            constants.wdExportFormatPDF,
                            Item=constants.wdExportDocumentWithMarkup,
                            CreateBookmarks=constants.wdExportCreateHeadingBookmarks)
    word.Quit(constants.wdDoNotSaveChanges)


if __name__ == '__main__':
    # 利用for循环+input语句可实现多目录下文件的批量转换
    dirname = r'D:\坚果云\Markdown\海南大学 大三课程笔记\网络工程'
    path(dirname, 2)
