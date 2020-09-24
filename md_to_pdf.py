# 待安装环境pandoc~~操作本质是os.system调用命令行执行该程序
# pip install pywin32
# 调用了word环境
import os
import time
from win32com.client import constants, gencache


# markdown文件转换模块  markdown——>docx
def path(dir_path):
    # os.walk输出当前目录下所有文件
    # x路径名， y文件夹名， z文件名
    for x, y, z in os.walk(dir_path):
        for filename in z:
            # 分割文件名和后缀名
            suffix_name = os.path.splitext(filename)[1]
            if suffix_name == '.md':
                file_path = x + '\\' + filename
                print('pandoc ' + '"' + file_path + '"' + ' -o ' + x + '\\' + os.path.splitext(filename)[0] + '.docx')
                os.system('pandoc ' + '"' + file_path + '"' + ' -o ' + '"' + x + '\\' + os.path.splitext(filename)[
                    0] + '.docx' + '"')


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


def batch(aim_path):
    for a, b, c in os.walk(aim_path):
        for name in c:
            suffix_name_docx = os.path.splitext(name)[1]
            prefix_name = os.path.splitext(name)[0]
            if suffix_name_docx == '.docx':
                filepath = a + '\\' + name
                outfilepath = a + '\\' + prefix_name
                createPdf(filepath, outfilepath)
                os.remove(filepath)


def pdf_file_move(aim):
    for a, b, c in os.walk(aim):
        for name in c:
            suffix_name_docx = os.path.splitext(name)[1]
            prefix_name = os.path.splitext(name)[0]
            if suffix_name_docx == '.docx':
                filepath = a + '\\' + name
                outfilepath = a + '\\' + prefix_name
                createPdf(filepath, outfilepath)
                os.remove(filepath)


if __name__ == '__main__':
    dirname = r'D:\坚果云\Markdown\海南大学 大三课程笔记\网络工程'
    # dirname = r'D:\坚果云\Markdown\海南大学 大三课程笔记\网络系统集成与综合布线'
    path(dirname)
    batch(dirname)