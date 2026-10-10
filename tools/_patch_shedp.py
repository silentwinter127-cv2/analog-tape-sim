p = "atsim/core/profiles.py"
s = open(p, encoding="utf-8").read()
anchor = '    g = "亮度电平"'
add = '''    g = "涂层脱落"
    tpe.define("shed_grain", "脱粉颗粒", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.7, source="参考素材发霉格：高频为正常格的 3 倍",
               help="磁粉脱落使磁头读到更粗糙的表面，重放信号带有宽带颗粒。"
                    "它与「粒子噪声底」的区别有两点：**强得多且更细**"
                    "（参考发霉格高频 0.116 而正常格 0.039；噪声底拉满只能到 0.035，"
                    "差六倍，所以那个样子根本做不出来），以及**只作用于亮度**——"
                    "涂层脱落会直接毁掉微弱的色度载波，颗粒若带上彩色就错了。"
                    "它与「色度增益」「消隐电平」共同构成发霉格的三要素："
                    "去饱和 + 变暗 + 高频颗粒。")
    ''' + anchor
if 'shed_grain' not in s:
    s = s.replace(anchor, add, 1)
    open(p, "w", encoding="utf-8").write(s)
    print("shed_grain parameter added")
else:
    print("already present")
