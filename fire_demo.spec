# -*- mode: python -*-

block_cipher = None


a = Analysis(['fire_demo.py'],
             pathex=['C:\\Users\\yoann\\PycharmProjects\\FireEffect'],
             binaries=[],
             datas=[('firepit.wav', '.'),
                    ('FireEffect.cp36-win_amd64.pyd', '.'),
                    ('FireEffect.cp38-win_amd64.pyd', '.'),
                    ('hsl.cp36-win_amd64.pyd',  '.'),
                    ('hsl.cp38-win_amd64.pyd',  '.'),
                    ('hsv.cp36-win_amd64.pyd',  '.'),
                    ('hsv.cp38-win_amd64.pyd',  '.'),
                    ('rand.cp36-win_amd64.pyd', '.'),
                    ('rand.cp38-win_amd64.pyd', '.')
                   ],

             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='fire_demo',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
