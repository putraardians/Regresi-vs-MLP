@echo off
echo Setting up Git repository for Regresi-vs-MLP project...

REM Check if Git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Git is not installed or not in PATH.
    echo Please install Git first:
    echo 1. Download from: https://git-scm.com/download/windows
    echo 2. Or use: winget install Git.Git
    echo 3. Or use: choco install git
    echo Then restart your terminal and run this script again.
    pause
    exit /b 1
)

echo Git found! Setting up repository...

REM Initialize Git repository
git init

REM Add all files
git add .

REM Make initial commit
git commit -m "Initial commit: Regresi vs MLP comparison project

- Added comprehensive model comparison between Linear Regression and MLP
- Included multiple MLP architecture experiments  
- Generated PDF reports with detailed analysis
- Added visualizations and summary files
- Complete project structure with documentation"

REM Set default branch to main
git branch -M main

REM Add remote origin (you may need to create the repository on GitHub first)
git remote add origin https://github.com/putraardians/Regresi-vs-MLP.git

echo Repository setup complete!
echo.
echo Next steps:
echo 1. Create repository on GitHub: https://github.com/new
echo 2. Repository name: Regresi-vs-MLP
echo 3. Make it public or private as needed
echo 4. Run: git push -u origin main
echo.
echo If remote repository already exists, run:
echo git push -u origin main
echo.
pause