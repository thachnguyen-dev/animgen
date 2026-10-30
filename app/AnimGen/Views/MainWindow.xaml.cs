using System.Windows;
using AnimGen.ViewModels;

namespace AnimGen.Views;

public partial class MainWindow : Window
{
    public MainViewModel Vm { get; }

    public MainWindow(MainViewModel vm)
    {
        Vm = vm;
        DataContext = vm;
        InitializeComponent();
        vm.SkeletonCheckRequested += (_, _) => CheckSkeleton();
    }

    private void CheckSkeleton_Click(object sender, RoutedEventArgs e) => CheckSkeleton();

    private void CheckSkeleton()
    {
        if (Vm.Api == null || Vm.Skeleton?.Image == null) return;
        try
        {
            var dlg = new SkeletonDialog(Vm.Api, Vm.Skeleton) { Owner = this };
            if (dlg.ShowDialog() == true && dlg.Reply != null) Vm.ApplySkeletonReply(dlg.Reply);
        }
        catch (InvalidOperationException ex)
        {
            MessageBox.Show(this, ex.Message, Title, MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private async void AddWeapon_Click(object sender, RoutedEventArgs e)
    {
        if (Vm.Api == null) return;
        var dlg = new WeaponDialog(Vm.Api) { Owner = this };
        dlg.ShowDialog();
        try
        {
            await Vm.ReloadWeaponsAsync(dlg.SavedWeaponId);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, Title, MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }
}
