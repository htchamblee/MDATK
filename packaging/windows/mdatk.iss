; Inno Setup script of the MDATK Windows installer (BETA).
;
;   ISCC.exe /DAppVersion=0.0.4 /DSourceDir=<dist\mdatk> /DOutputDir=<dist> ^
;            /DOutputName=MDATK-0.0.4-windows-x64-setup /DIconFile=<mdatk.ico> packaging\windows\mdatk.iss
;
; packaging/windows/build_windows.sh calls it with these values. Installs per user by default
; (no administrator rights needed); "Install for all users" is offered in the setup dialog.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #error SourceDir (the PyInstaller bundle folder dist\mdatk) must be given with /DSourceDir=...
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif
#ifndef OutputName
  #define OutputName "MDATK-" + AppVersion + "-windows-x64-setup"
#endif

[Setup]
AppId={{6F1D5C3A-2B7E-4C1F-9A55-8D3E2B4C7A10}
AppName=MDATK
AppVersion={#AppVersion}
AppVerName=MDATK {#AppVersion} (beta)
AppPublisher=MDATK contributors
AppPublisherURL=https://github.com/htchamblee/MDATK
AppSupportURL=https://github.com/htchamblee/MDATK/issues
AppComments=Molecular Dynamic Analysis Tool Kit
DefaultDirName={autopf}\MDATK
DefaultGroupName=MDATK
DisableProgramGroupPage=yes
LicenseFile={#SourceDir}\LICENSE.txt
InfoBeforeFile={#SourceDir}\README.txt
OutputDir={#OutputDir}
OutputBaseFilename={#OutputName}
#ifdef IconFile
SetupIconFile={#IconFile}
#endif
UninstallDisplayIcon={app}\mdatk-gui.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ChangesEnvironment=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "addtopath"; Description: "Add the mdatk command line program to PATH"; GroupDescription: "Command line:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\MDATK"; Filename: "{app}\mdatk-gui.exe"
Name: "{group}\MDATK command prompt"; Filename: "{cmd}"; Parameters: "/k ""{app}\mdatk.exe"" --help"; \
    WorkingDir: "{%USERPROFILE}"
Name: "{group}\Licences"; Filename: "{app}\licenses\THIRD_PARTY_NOTICES.txt"
Name: "{group}\{cm:UninstallProgram,MDATK}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\MDATK"; Filename: "{app}\mdatk-gui.exe"; Tasks: desktopicon

[Registry]
; per-user or all-users PATH entry, depending on the install mode
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Tasks: addtopath; Check: (not IsAdminInstallMode) and NeedsAddPath(ExpandConstant('{app}'))
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; \
    ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Tasks: addtopath; Check: IsAdminInstallMode and NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\mdatk-gui.exe"; Description: "{cm:LaunchProgram,MDATK}"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsAddPath(Dir: string): Boolean;
var
  Root: Integer;
  Key, Paths: string;
begin
  if IsAdminInstallMode then begin
    Root := HKEY_LOCAL_MACHINE;
    Key := 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';
  end else begin
    Root := HKEY_CURRENT_USER;
    Key := 'Environment';
  end;
  if not RegQueryStringValue(Root, Key, 'Path', Paths) then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') = 0;
end;

procedure RemovePath(Dir: string);
var
  Root: Integer;
  Key, Paths: string;
  P: Integer;
begin
  if IsAdminInstallMode then begin
    Root := HKEY_LOCAL_MACHINE;
    Key := 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';
  end else begin
    Root := HKEY_CURRENT_USER;
    Key := 'Environment';
  end;
  if not RegQueryStringValue(Root, Key, 'Path', Paths) then
    exit;
  P := Pos(';' + Uppercase(Dir), Uppercase(Paths));
  if P > 0 then begin
    Delete(Paths, P, Length(Dir) + 1);
    RegWriteExpandStringValue(Root, Key, 'Path', Paths);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemovePath(ExpandConstant('{app}'));
end;
