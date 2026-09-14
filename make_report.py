"""从真实 CSV / JSON 制作图表及三页报告。"""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'report'; out.mkdir(exist_ok=True)
    names=['baseline','label_smoothing']
    results=[json.loads((ROOT/f'runs/{n}/evaluation/results.json').read_text()) for n in names]
    histories=[list(csv.DictReader((ROOT/f'runs/{n}/history.csv').open())) for n in names]
    configs=[json.loads((ROOT/f'runs/{n}/config.json').read_text()) for n in names]
    fig,axes=plt.subplots(1,3,figsize=(12,3.0))
    for name,h,color in zip(['Baseline','LS 0.1'],histories,['#236e96','#d17a29']):
        ep=[int(r['epoch']) for r in h]
        axes[0].plot(ep,[float(r['train_ce']) for r in h],color=color,label=name+' train')
        axes[0].plot(ep,[float(r['val_loss']) for r in h],color=color,ls='--',label=name+' val')
        axes[1].plot(ep,[float(r['train_accuracy']) for r in h],color=color,label=name+' train')
        axes[1].plot(ep,[float(r['val_accuracy']) for r in h],color=color,ls='--',label=name+' val')
        axes[2].plot(ep,[float(r['train_loss']) for r in h],color=color,label=name)
    for ax,title in zip(axes,['Common cross-entropy','Top-1 accuracy (%)','Training objective loss']):
        ax.set(title=title,xlabel='Epoch'); ax.grid(alpha=.2); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=200); plt.close(fig)
    chosen=max(range(2),key=lambda i:results[i]['val_metrics']['top1'])
    selected=results[chosen]; selected_name=names[chosen]
    matrix=np.loadtxt(ROOT/f'runs/{selected_name}/evaluation/confusion_matrix.csv',delimiter=',')
    split=json.loads((ROOT/'runs/split.json').read_text())
    breed_names=split['classes']
    pair_classes=list(dict.fromkeys(x for pair in selected['confusion_pairs'] for x in pair[1:3]))
    indices=[breed_names.index(x) for x in pair_classes]
    local=matrix[np.ix_(indices,indices)]
    fig,ax=plt.subplots(figsize=(7,3.4))
    ax.imshow(local,cmap='Blues')
    ax.set_xticks(range(len(indices)),pair_classes,rotation=22,ha='right',fontsize=7)
    ax.set_yticks(range(len(indices)),pair_classes,fontsize=7)
    ax.set(xlabel='Predicted breed',ylabel='True breed')
    for i in range(len(indices)):
        for j in range(len(indices)):
            ax.text(j,i,str(int(local[i,j])),ha='center',va='center',fontsize=8,
                    color='white' if local[i,j]>local.max()/2 else '#182b3a')
    fig.tight_layout(); fig.savefig(out/'confusion_detail.png',dpi=200); plt.close(fig)
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    body=ParagraphStyle('body',fontName='STSong-Light',fontSize=10,leading=16,textColor=colors.HexColor('#233341'),spaceAfter=7)
    heading=ParagraphStyle('heading',parent=body,fontSize=14,leading=20,spaceBefore=9,spaceAfter=9,textColor=colors.HexColor('#155878'))
    title=ParagraphStyle('title',parent=heading,fontSize=19,leading=26)
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=12)
    story=[]; prose=[]
    def p(s,style=body): story.append(Paragraph(s,style)); prose.append(s.replace('<br/>','\n'))
    def img(path,w,h): story.append(Image(str(path),width=w,height=h))
    p('基于深度学习的牛津宠物<br/>细粒度分类实验报告',title)
    p('王诗雯  |  W125301282  |  ResNet-18 / 单变量消融',small)
    repo=(ROOT/'repository.txt').read_text().strip() if (ROOT/'repository.txt').exists() else '仓库发布状态：尚未上传'
    p(repo,small)
    p('1  任务与实验设置',heading)
    sizes=configs[0]['split_sizes']
    p(f'我使用 Oxford-IIIT Pet 的 37 个宠物品种进行细粒度分类。实际载入 {split["total"]} 张图像；我合并官方 trainval 与 test，使用 seed=42 分层划分为训练 {sizes["train"]} 张、验证 {sizes["val"]} 张、测试 {sizes["test"]} 张。三部分索引互斥且均覆盖全部类别。本实验采用任务书要求的自定义划分，不与官方测试划分下的成绩直接比较。')
    p('我将 ImageNet 预训练 ResNet-18 的分类头替换为 37 类线性层，采用全参数微调。训练输入依次执行 Resize(256)、RandomCrop(224)、随机水平翻转及 ImageNet 标准化；验证和测试采用固定中心裁剪。输入 batch 为 [32,3,224,224]，最后卷积特征为 [32,512,7,7]。')
    p(f'我在 {configs[0]["gpu"] or "CPU"} 上运行 PyTorch {configs[0]["torch"]}。两组均使用 AdamW（lr=0.0001，weight decay=0.01）、batch size=32、10 轮、相同种子和数据顺序；不使用学习率调度及提前终止。仅将交叉熵的 Label Smoothing 系数由 0 改为 0.1。')
    p('2  实验结果与消融分析',heading)
    rows=[['实验','测试 Top-1','测试 Top-5','Macro-F1','最佳轮次','训练耗时']]
    for i,label in enumerate(['Baseline','LS 0.1']):
        m=results[i]['metrics']; rows.append([label,f'{m["top1"]:.2f}%',f'{m["top5"]:.2f}%',f'{m["macro_f1"]:.4f}',str(results[i]['checkpoint_epoch']),f'{float(histories[i][-1]["seconds"])/60:.1f} min'])
    prose.append('\n'.join(['| '+' | '.join(row)+' |' for row in [rows[0], ['---']*6]+rows[1:]]))
    with (out/'experiment_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
    table=Table(rows,colWidths=[78,78,78,78,68,78],hAlign='LEFT')
    table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'STSong-Light'),('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6f0f5')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#155878')),('BOTTOMPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#a8c3d1'))]))
    story.append(table); story.append(Spacer(1,9))
    d=results[1]['metrics']['top1']-results[0]['metrics']['top1']
    f1=results[1]['metrics']['macro_f1']-results[0]['metrics']['macro_f1']
    p(f'我仅按验证 Top-1 选取各组最佳 checkpoint：Baseline 为 {results[0]["val_metrics"]["top1"]:.2f}%，LS 0.1 为 {results[1]["val_metrics"]["top1"]:.2f}%。最终测试中，平滑组相对基线的 Top-1 变化为 {d:+.2f} 个百分点，Macro-F1 变化为 {f1:+.4f}。我将其视为本次固定种子下的观察，不能据此认定统计显著。耗时包含数据加载器初始化，不含下载、测试和报告生成，不作为纯计算性能基准。')
    p(f'我也检查其余指标：两组测试 Top-5 分别为 {results[0]["metrics"]["top5"]:.2f}% 和 {results[1]["metrics"]["top5"]:.2f}%，普通测试交叉熵分别为 {results[0]["metrics"]["loss"]:.3f} 和 {results[1]["metrics"]["loss"]:.3f}。我分别报告各项指标，不将单项改善等同于所有指标同时提升。')
    story.append(PageBreak())
    p('2  收敛曲线与结果解释（续）',heading)
    img(out/'training_curves.png',490,122.5)
    p('图 1  左：统一普通交叉熵；中：训练与验证 Top-1；右：各组实际优化目标。虚线表示验证集。',small)
    for label,h in zip(['Baseline','LS 0.1'],histories):
        p(f'我的 {label} 验证准确率由第 1 轮的 {float(h[0]["val_accuracy"]):.2f}% 变为第 10 轮的 {float(h[-1]["val_accuracy"]):.2f}%；普通验证交叉熵由 {float(h[0]["val_loss"]):.3f} 变为 {float(h[-1]["val_loss"]):.3f}。我保留验证准确率最高轮次，避免将最后一轮默认视为最优。')
    p('我额外记录训练集的普通交叉熵 train_ce，因为加入标签平滑后的目标损失与基线损失数值不可直接比较。训练集使用随机增强，训练指标又累积自一轮中不断更新的参数，因而训练与验证差距只能辅助判断过拟合，不能单凭差距定论。')
    p('基线后期训练准确率持续上升，而验证准确率未同步改善，出现过拟合迹象。平滑组的验证 Top-1 更高，但训练与验证仍存在差距；仅凭这组曲线，我不能认定过拟合已被消除。')
    p('3  混淆品种与错误分析',heading)
    p(f'我按验证成绩选取 {selected_name} 展示案例，测试集不参与配置选择。以下按双向误判总数排序，列出最易混淆的三对品种；括号内依次是前者被判为后者、后者被判为前者的样本数。')
    for total,a,b,ab,ba in selected['confusion_pairs']:
        p(f'{a} / {b}：合计 {total} 次（{ab} / {ba}）。',small)
    img(out/'confusion_detail.png',410,199)
    p('图 2  高频混淆类别局部矩阵（计数）。完整 37 类行归一化矩阵与原始计数另存于 evaluation 文件夹；局部图只显示所选类别之间的预测。',small)
    story.append(PageBreak())
    p('3  Grad-CAM 与案例复盘（续）',heading)
    for kind,label in [('correct','预测正确'),('wrong','预测错误')]:
        if kind in selected['cases']:
            img(ROOT/f'runs/{selected_name}/evaluation/gradcam_{kind}.png',410,205)
            case=selected['cases'][kind]
            p(f'图 {3 if kind=="correct" else 4}  {label}案例：真实类别 {case["true"]}，预测类别 {case["pred"]}。样本为固定测试顺序中的第一个{label}样本。',small)
    note_path=ROOT/'analysis_notes.json'
    notes=json.loads(note_path.read_text(encoding='utf-8')) if note_path.exists() else {}
    p((notes.get('cam') if notes.get('cases') == selected['cases'] else None) or '我对预测类别计算 Grad-CAM，以最后残差阶段特征的空间平均梯度作为通道权重。热力图反映当前预测分数对特征区域的敏感程度，不等同于分割标注，也不能单独证明模型识别了特定部位。',small)
    p('4  AI 辅助编程声明与排错复盘',heading)
    p('我使用 AI 辅助生成数据管线、训练循环、Grad-CAM 及报告汇总代码，指标和图片均来自实际运行。训练中每批先清零梯度，验证采用 eval() 与 no_grad()，分类头直接输出 logits；Grad-CAM 独立保留梯度。',small)
    p(notes.get('bug','环境排错记录：系统默认 Python 无法正常创建带 pip 的虚拟环境，出现 ensurepip 失败；我改用独立 Python 3.12 环境完成依赖安装。此问题属于环境配置，不冒充模型代码缺陷。'),small)
    p('参考：Oxford-IIIT Pet 数据集（robots.ox.ac.uk/~vgg/data/pets/）；Torchvision ResNet-18 官方文档（docs.pytorch.org/vision/）。完整代码、配置、逐轮 CSV、TensorBoard 和逐样本预测随项目交付。',small)
    def footer(canvas,doc):
        canvas.setFont('STSong-Light',8); canvas.setFillColor(colors.HexColor('#607787'))
        canvas.drawString(44,26,'王诗雯  /  W125301282  /  Oxford-IIIT Pet')
        canvas.drawRightString(A4[0]-44,26,f'{doc.page}')
    target=out/'【考核】王诗雯_W125301282_宠物分类.pdf'
    SimpleDocTemplate(str(target),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=34,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    (out/'报告正文.md').write_text('\n\n'.join(prose),encoding='utf-8')
    print(target)

if __name__=='__main__': main()
