
# Install git submodules

git submodule init
git submodule update

cd ./submodules/cotracker
git checkout 8d364031971f6b3efec945dd15c468a183e58212
cd ../..

cp -rv ./submodules/cotracker/cotracker ./cotracker
cp -rv ./submodules/r2d2 ./DBoW/r2d2
